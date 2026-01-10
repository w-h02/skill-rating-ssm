import particles
import particles.state_space_models as ssm
import particles.distributions as dists
from particles import SMC
import numpy as np
import math
from scipy.stats import norm
from scipy import optimize
import matplotlib.pyplot as plt
import pandas as pd
from tqdm import tqdm

# ssm model
class SkillRatingModel(ssm.StateSpaceModel):
    def __init__(self, sigma_sq):
        super().__init__()
        self.sigma_sq = sigma_sq

    def PX0(self):
        """Initial skill distribution"""
        return dists.Normal(loc=0., scale=1.)

    def PX(self, t, xp):
        """Skill transition: x_t | x_{t-1}
            xp: Previous state"""
        return dists.Normal(loc = xp, scale = np.sqrt(self.sigma_sq))
    

# Filter
class PairwiseSkillFilter():
    """
    Manages independent particle filters for each team and performs
    pairwise updates when teams play each other.
    
    This implements the factorization approximation from Section 3.4.1.

    ATTRIBUTES:
        num_teams: Number of teams
        
        num_particles: Number of particles per team (M)
        
        sigma_sq: Skill evolution variance
        
        sigma_obs_sq: Observation noise variance
        
        draw_threshold: Threshold δ for modeling draws (optional, default 0.5)
        
        filters: Dictionary mapping team_id -> particles.SMC object
            Each SMC object is an independent particle filter for one team
        
        particles: Dictionary mapping team_id -> array of particles (shape: num_particles,)
        
        weights:  Dictionary mapping team_id -> array of weights (shape: num_particles,)
        
        skill_history: List of dictionaries, where skill_history[t][team_id] = estimated skill at time t
        
        log_likelihood: Running total of log p(y_{1:T} | θ)
    """
    def __init__(self, num_teams, num_particles, sigma_sq, sigma_obs_sq, draw_threshold=0.5):
        
        self.num_teams = num_teams
        self.num_particles = num_particles
        self.sigma_sq = sigma_sq
        self.sigma_obs_sq = sigma_obs_sq
        self.draw_threshold = draw_threshold
        
        self.models = {}     # SkillRatingModel per team
        self.particles = {}  # particles per team
        self.weights = {}    # weights per team

        for team_id in range(num_teams):
            model = SkillRatingModel(sigma_sq = self.sigma_sq)
            self.models[team_id] = model
            # Initial distribution and weights
            initial_dist = model.PX0()
            self.particles[team_id] = initial_dist.rvs(size = self.num_particles)
            self.weights[team_id] = np.ones(self.num_particles) / self.num_particles
        
        self.skill_history = []
        self.log_likelihood = 0.0

    def run_filter(self, matches):
        """
        INPUT:
           matches: list of match dictionaries
                      each match has: {team_A_id, team_B_id, outcome, date}

        OUTPUT:
            particles, weights, skill_history, log_likelihood
        """
        
        for match in tqdm(matches, desc="Processing matches", leave=False):
            team_home = match['team_h']
            team_away = match['team_a']
            outcome = match['outcome']

            # PREDICT STEP: propagate all teams forward
            self.predict_all_teams()

            # UPDATE STEP: update only teams home and away
            match_log_lik = self.pairwise_update(team_home, team_away, outcome)
            self.log_likelihood += match_log_lik

            # RESAMPLE if needed
            self.resample(team_home)
            self.resample(team_away)

            # Get current skill estimates
            current_skills = self.get_current_skills()
            self.skill_history.append(current_skills)

        return self.particles, self.weights, self.skill_history, self.log_likelihood
    
    def predict_all_teams(self):
        for team_id in range(self.num_teams):
            noise = np.random.normal(
                loc=0.0,
                scale=np.sqrt(self.sigma_sq),
                size=self.num_particles
            )
            self.particles[team_id] += noise

    
    def pairwise_update(self, team_home, team_away, outcome):
        # Update weights for teams home and away based on match outcome

        particles_h, weights_h = self.particles[team_home], self.weights[team_home]
        particles_a, weights_a = self.particles[team_away], self.weights[team_away]

        likelihoods = np.zeros((self.num_particles, self.num_particles))

        for i in range(self.num_particles):
            for j in range(self.num_particles):
                skill_diff = particles_h[i] - particles_a[j]
                # Compute P(outcome | skill_A[i], skill_B[j])
                prob = self.compute_match_probability(skill_diff, outcome)
                # Weight by current particle weights
                likelihoods[i, j] = prob * weights_h[i] * weights_a[j]

        total_likelihood = np.sum(likelihoods)
        if total_likelihood == 0:
            return float('-inf')
        
        # computing weights
        new_weights_h = np.sum(likelihoods, axis=1)  # sum over team away
        new_weights_a = np.sum(likelihoods, axis=0)  # sum over team home 

        # Normalizing weights
        new_weights_h /= sum(new_weights_h)
        new_weights_a /= sum(new_weights_a)

        # Updating weights
        self.weights[team_home] = new_weights_h
        self.weights[team_away] = new_weights_a

        # Match log likelihood
        match_log_likelihood = np.log(total_likelihood)
        return match_log_likelihood
    

    def compute_match_probability(self, skill_diff, outcome):
        """
        Compute P(outcome | skill_diff) using the Thurstone-Mosteller model
        Input: skill_diff : Skill difference between the teams, 
                outcome :  1 for win home, 0 for draw, -1 for win away
        Output: probability of win or draw depending on the outcome
        """

        z = skill_diff / (np.sqrt(2 * self.sigma_obs_sq))
        if outcome == 1:
            prob = norm.cdf(z)
        elif outcome == -1:
            prob = norm.cdf(-z)
        else:
            z_upper = (self.draw_threshold - skill_diff) / (np.sqrt(2 * self.sigma_obs_sq))
            z_lower = (-self.draw_threshold - skill_diff) / (np.sqrt(2 * self.sigma_obs_sq))
            prob = norm.cdf(z_upper) - norm.cdf(z_lower)

        return max(prob, 1e-10)
    
    def resample(self, team_id):
        """
        Check ESS and resample if necessary
        """
        weights = self.weights[team_id]
        ess = 1.0 / np.sum(weights ** 2)

        if ess < self.num_particles / 2:
            particles = self.particles[team_id]
            
            # Systematic resampling
            indices = self.systematic_resampling(weights)
            
            # Update particles and reset weights
            self.particles[team_id] = particles[indices]
            self.weights[team_id] = np.ones(self.num_particles) / self.num_particles
    
    
    def systematic_resampling(self, weights):
        """Standard systematic resampling algorithm"""
        N = len(weights)
        cumsum = np.cumsum(weights)

        # Random starting Point
        u = np.random.uniform(0, 1/N)
        indices = np.zeros(N, dtype=int)

        for i in range(N):
            u_i = u + i/N
            indices[i] = np.searchsorted(cumsum, u_i)

        return indices
    
    def get_current_skills(self):
        """Get weighted mean skill estimate for each team"""
        current_skills = {}
        for team_id in range(self.num_teams):
            particles = self.particles[team_id]
            weights = self.weights[team_id]
            
            # Weighted mean
            skill_estimate = np.sum(particles * weights)
            current_skills[team_id] = skill_estimate
        
        return current_skills
    

# EM Algorithm

class EM_Estimator():
    def __init__(self, matches, num_teams, num_particles=1000, 
                     max_iterations=50, tolerance=1e-4, draw_threshold = 0.5):
        self.matches = matches
        self.num_teams = num_teams
        self.num_particles = num_particles
        self.max_iterations = max_iterations
        self.tolerance = tolerance
        self.draw_threshold = draw_threshold
    
    def run_EM(self):
        # Initialize parameters
        sigma_sq = 0.1  # initial guess
        sigma_obs_sq = 1.0  # initial guess
        
        param_history = []
        log_likelihood_history = []
        
        print("Starting EM algorithm...")
        print(f"Initial parameters: sigma_sq = {sigma_sq:.4f}, sigma_obs_sq = {sigma_obs_sq:.4f}")
        
        for iteration in tqdm(range(self.max_iterations), desc="EM Iterations"):
            print(f"\n{'=' * 60}")
            print(f"EM Iteration: {iteration + 1}/{self.max_iterations}")
            print(f"{'=' * 60}")

            # ========== E-step ========== 
            # Run particle filter with current parameters
            pf = PairwiseSkillFilter(
                num_teams = self.num_teams,
                num_particles = self.num_particles,
                sigma_sq = sigma_sq,
                sigma_obs_sq = sigma_obs_sq,
                draw_threshold = self.draw_threshold
            )
            
            print("Running particle filter...")
            particles, weights, skill_history, log_lik = pf.run_filter(self.matches)
            
            param_history.append([sigma_sq, sigma_obs_sq])
            log_likelihood_history.append(log_lik)
            
            print(f"Log-likelihood: {log_lik:.4f}")
            print(f"Current parameters: sigma_sq = {sigma_sq:.6f}, sigma_obs_sq = {sigma_obs_sq:.6f}")
            
            # ========== M-STEP ==========
            print("Updating parameters...")
            
            sigma_sq_new = self.update_sigma_sq(skill_history)
            sigma_obs_sq_new = self.update_sigma_obs_sq(self.matches, skill_history)
            
            print(f"New parameters: sigma_sq = {sigma_sq_new:.6f}, sigma_obs_sq = {sigma_obs_sq_new:.6f}")
            
            # Check convergence
            param_change = abs(sigma_sq_new - sigma_sq) + abs(sigma_obs_sq_new - sigma_obs_sq)
            print(f"Parameter change: {param_change:.8f} (tolerance: {self.tolerance})")
            
            if param_change < self.tolerance:
                print(f"\n{'='*60}")
                print(f"Converged at iteration {iteration + 1}!")
                print(f"{'='*60}")
                # Update to final values
                sigma_sq = sigma_sq_new
                sigma_obs_sq = sigma_obs_sq_new
                # Append final values
                param_history.append([sigma_sq, sigma_obs_sq])
                
                # Run one more time to get final log-likelihood
                pf_final = PairwiseSkillFilter(
                    num_teams=self.num_teams,
                    num_particles=self.num_particles,
                    sigma_sq=sigma_sq,
                    sigma_obs_sq=sigma_obs_sq,
                    draw_threshold=self.draw_threshold
                )
                _, _, _, final_log_lik = pf_final.run_filter(self.matches)
                log_likelihood_history.append(final_log_lik)
                break
            
            # Update parameters
            sigma_sq = sigma_sq_new
            sigma_obs_sq = sigma_obs_sq_new
        else:
            print(f"\n{'='*60}")
            print(f"Reached maximum iterations ({self.max_iterations})")
            print(f"{'='*60}")
        
        print(f"\nFinal estimates:")
        print(f"  sigma_sq = {sigma_sq:.6f}")
        print(f"  sigma_obs_sq = {sigma_obs_sq:.6f}")
        print(f"  Final log-likelihood = {log_likelihood_history[-1]:.4f}")
        
        return sigma_sq, sigma_obs_sq, param_history, log_likelihood_history
    
    def update_sigma_sq(self, skill_history):
        """
        M-step update for sigma_squared.
        
        Mathematical formula: sigma_new = (1/NT) Σ_t Σ_i (s_i^(t) - s_i^(t-1))²
        
        Parameters:
        -----------
        skill_history : list of dict
            List where skill_history[t][team_id] = skill at time t
        
        Returns:
        --------
        sigma_sq_new : float
            Updated estimate of skill evolution variance
        """
        num_matches = len(skill_history)
        
        if num_matches < 2:
            print("Warning: Not enough data to estimate sigma_sq, returning default")
            return 0.1
        
        total_squared_change = 0.0
        count = 0
        
        for team_id in range(self.num_teams):
            for t in range(1, num_matches):
                skill_t = skill_history[t][team_id]
                skill_t_prev = skill_history[t-1][team_id]
                
                squared_change = (skill_t - skill_t_prev) ** 2
                total_squared_change += squared_change
                count += 1
        
        if count == 0:
            print("Warning: No skill changes to compute, returning default sigma_sq")
            return 0.1
        
        sigma_sq_new = total_squared_change / count
        
        # Ensure positivity and reasonable bounds
        sigma_sq_new = max(sigma_sq_new, 1e-6)  # Lower bound
        sigma_sq_new = min(sigma_sq_new, 10.0)  # Upper bound
        
        return sigma_sq_new
    
    def update_sigma_obs_sq(self, matches, skill_history):
        """
        M-step update for sigma_squared_obs.
        --> we use numerical optimization.
        
        Parameters:
        -----------
        matches : list
            List of matches
        skill_history : list of dict
            Skill estimates from E-step
        
        Returns:
        --------
        sigma_obs_sq_new : float
            Updated estimate of observation noise variance
        """
        
        def objective_function(sigma_obs_sq_candidate):
            """
            Compute expected log-likelihood given current skill estimates.
            
            We want to MAXIMIZE this, so we'll minimize its negative.
            """
            if sigma_obs_sq_candidate <= 0:
                return -np.inf
            
            total_log_lik = 0.0
            
            for match_idx, match in enumerate(matches):
                team_h = match['team_h']
                team_a = match['team_a']
                outcome = match['outcome']
                
                # Get skill estimates at this match time
                skill_h = skill_history[match_idx][team_h]
                skill_a = skill_history[match_idx][team_a]
                skill_diff = skill_h - skill_a
                
                # Compute P(outcome | skill_diff, sigma_obs_sq_candidate)
                z = skill_diff / np.sqrt(2 * sigma_obs_sq_candidate)
                
                if outcome == 1:  # Home wins
                    prob = norm.cdf(z)
                elif outcome == -1:  # Away wins
                    prob = norm.cdf(-z)
                else:  # Draw (outcome == 0)
                    z_upper = (self.draw_threshold - skill_diff) / np.sqrt(2 * sigma_obs_sq_candidate)
                    z_lower = (-self.draw_threshold - skill_diff) / np.sqrt(2 * sigma_obs_sq_candidate)
                    prob = norm.cdf(z_upper) - norm.cdf(z_lower)
                
                # Avoid log(0)
                prob = max(prob, 1e-10)
                total_log_lik += np.log(prob)
            
            return total_log_lik
        
        # We minimize the negative to maximize the original
        result = optimize.minimize_scalar(
            lambda x: -objective_function(x),
            bounds=(0.01, 10.0),
            method='bounded',
            options={'xatol': 1e-6}
        )
        
        if not result.success:
            print("Warning: Optimization for sigma_obs_sq did not converge properly")
            print(f"  Message: {result.message}")
        
        sigma_obs_sq_new = result.x
        
        # Ensure reasonable bounds
        sigma_obs_sq_new = max(sigma_obs_sq_new, 0.01)
        sigma_obs_sq_new = min(sigma_obs_sq_new, 10.0)
        
        return sigma_obs_sq_new
    

class Visualizer:
    """
    Creates plots and visualizations.
    """
    
    @staticmethod
    def create_figure_3(log_lik_grid, sigma_sq_vals, sigma_obs_sq_vals,
                       em_sigma_sq, em_sigma_obs_sq, param_history):
        """
        Create the log-likelihood surface plot (Figure 3).
        
        Parameters:
        -----------
        log_lik_grid : 2D array
            Grid of log-likelihoods (grid_size x grid_size)
        sigma_sq_vals : 1D array
            sigma_squared values (x-axis)
        sigma_obs_sq_vals : 1D array
            sigma_squared_obs values (y-axis)
        em_sigma_sq : float
            Final EM estimate of σ²
        em_sigma_obs_sq : float
            Final EM estimate of sigma_squared_obs
        param_history : list
            List of [σ², sigma_squared_obs] from each EM iteration
        """
        fig, ax = plt.subplots(figsize=(10, 8))
        
        # Create meshgrid for contour plot
        X, Y = np.meshgrid(sigma_sq_vals, sigma_obs_sq_vals)
        
        # Contour plot (filled)
        contour = ax.contourf(X, Y, log_lik_grid.T, levels=20, cmap='viridis')
        plt.colorbar(contour, label='Log-Likelihood', ax=ax)
        
        # Contour lines
        ax.contour(X, Y, log_lik_grid.T, levels=10, colors='white', 
                  alpha=0.3, linewidths=0.5)
        
        # Mark EM estimate
        ax.plot(em_sigma_sq, em_sigma_obs_sq, 'r*', markersize=20, 
               label='EM Estimate', markeredgecolor='white', markeredgewidth=1.5)
        
        # Plot EM convergence path
        if param_history is not None and len(param_history) > 0:
            path_sigma_sq = [p[0] for p in param_history]
            path_sigma_obs_sq = [p[1] for p in param_history]
            
            # Plot the path
            ax.plot(path_sigma_sq, path_sigma_obs_sq, 'w--', 
                   linewidth=2, label='EM Path', alpha=0.8)
            
            # Mark starting point
            ax.plot(path_sigma_sq[0], path_sigma_obs_sq[0], 'go', 
                   markersize=10, label='EM Start', markeredgecolor='white', 
                   markeredgewidth=1.5)
            
            # Optionally, add arrows to show direction
            for i in range(len(path_sigma_sq) - 1):
                ax.annotate('', 
                           xy=(path_sigma_sq[i+1], path_sigma_obs_sq[i+1]),
                           xytext=(path_sigma_sq[i], path_sigma_obs_sq[i]),
                           arrowprops=dict(arrowstyle='->', color='white', 
                                         lw=1, alpha=0.5))
        
        ax.set_xlabel('σ² (Skill Evolution Variance)', fontsize=12)
        ax.set_ylabel('σ²_obs (Observation Noise Variance)', fontsize=12)
        ax.set_title('Log-Likelihood Surface - Football Data', fontsize=14, fontweight='bold')
        ax.legend(loc='best', fontsize=10)
        ax.grid(True, alpha=0.3, linestyle='--', linewidth=0.5)
        
        plt.tight_layout()
        plt.savefig('figure_3_football.png', dpi=300, bbox_inches='tight')
        print("Figure 3 saved as 'figure_3_football.png'")
        plt.show()
    
    @staticmethod
    def plot_skill_trajectories(skill_history, teams, team_to_id, 
                                teams_to_plot=None, num_teams_plot=10):
        """
        Plot skill evolution over time for selected teams.
        
        Parameters:
        -----------
        skill_history : list of dict
            List where skill_history[t][team_id] = skill at time t
        teams : list
            List of team names
        team_to_id : dict
            Mapping from team name to team ID
        teams_to_plot : list, optional
            Specific teams to plot (or None for top teams)
        num_teams_plot : int
            How many teams to plot if teams_to_plot is None
        """
        if teams_to_plot is None:
            # Select teams with highest final skill
            final_skills = skill_history[-1]
            top_team_ids = sorted(final_skills.keys(), 
                                 key=lambda x: final_skills[x], 
                                 reverse=True)[:num_teams_plot]
            teams_to_plot = [teams[tid] for tid in top_team_ids]
        
        fig, ax = plt.subplots(figsize=(12, 8))
        
        # Use a colormap for different teams
        colors = plt.cm.tab10(np.linspace(0, 1, len(teams_to_plot)))
        
        for idx, team_name in enumerate(teams_to_plot):
            team_id = team_to_id[team_name]
            
            # Extract skill trajectory
            trajectory = [skill_history[t][team_id] for t in range(len(skill_history))]
            
            ax.plot(trajectory, label=team_name, linewidth=2, color=colors[idx])
        
        ax.set_xlabel('Match Number', fontsize=12)
        ax.set_ylabel('Skill Level', fontsize=12)
        ax.set_title('Team Skill Evolution Over Time', fontsize=14, fontweight='bold')
        ax.legend(bbox_to_anchor=(1.05, 1), loc='upper left', fontsize=10)
        ax.grid(True, alpha=0.3, linestyle='--', linewidth=0.5)
        ax.axhline(y=0, color='black', linestyle='-', linewidth=0.8, alpha=0.3)
        
        plt.tight_layout()
        plt.savefig('skill_trajectories.png', dpi=300, bbox_inches='tight')
        print("Skill trajectories saved as 'skill_trajectories.png'")
        plt.show()
    
    @staticmethod
    def plot_em_convergence(log_likelihood_history, param_history):
        """
        Plot EM convergence: log-likelihood and parameter evolution.
        
        Parameters:
        -----------
        log_likelihood_history : list
            Log-likelihood values at each EM iteration
        param_history : list
            List of [σ², σ²_obs] at each EM iteration
        """
        fig, axes = plt.subplots(2, 1, figsize=(10, 8))
        
        iterations = range(len(log_likelihood_history))
        
        # Plot log-likelihood
        axes[0].plot(iterations, log_likelihood_history, 'b-o', linewidth=2, markersize=6)
        axes[0].set_xlabel('EM Iteration', fontsize=11)
        axes[0].set_ylabel('Log-Likelihood', fontsize=11)
        axes[0].set_title('EM Convergence: Log-Likelihood', fontsize=12, fontweight='bold')
        axes[0].grid(True, alpha=0.3, linestyle='--')
        
        # Plot parameters
        if param_history is not None and len(param_history) > 0:
            param_iterations = range(len(param_history))
            sigma_sq_vals = [p[0] for p in param_history]
            sigma_obs_sq_vals = [p[1] for p in param_history]
            
            ax2 = axes[1]
            ax2.plot(param_iterations, sigma_sq_vals, 'r-o', linewidth=2, 
                    markersize=6, label='σ² (skill evolution)')
            ax2.set_xlabel('EM Iteration', fontsize=11)
            ax2.set_ylabel('σ²', fontsize=11, color='r')
            ax2.tick_params(axis='y', labelcolor='r')
            ax2.grid(True, alpha=0.3, linestyle='--')
            
            # Second y-axis for sigma_obs_sq
            ax3 = ax2.twinx()
            ax3.plot(param_iterations, sigma_obs_sq_vals, 'b-s', linewidth=2, 
                    markersize=6, label='σ²_obs (observation noise)')
            ax3.set_ylabel('σ²_obs', fontsize=11, color='b')
            ax3.tick_params(axis='y', labelcolor='b')
            
            # Combined legend
            lines1, labels1 = ax2.get_legend_handles_labels()
            lines2, labels2 = ax3.get_legend_handles_labels()
            ax2.legend(lines1 + lines2, labels1 + labels2, loc='best', fontsize=10)
            
            axes[1].set_title('EM Convergence: Parameters', fontsize=12, fontweight='bold')
        
        plt.tight_layout()
        plt.savefig('em_convergence.png', dpi=300, bbox_inches='tight')
        print("EM convergence plot saved as 'em_convergence.png'")
        plt.show()
    
    @staticmethod
    def plot_final_skill_ranking(skill_history, teams):
        """
        Plot final skill ranking of all teams.
        
        Parameters:
        -----------
        skill_history : list of dict
            Skill estimates over time
        teams : list
            List of team names
        """
        # Get final skills
        final_skills = skill_history[-1]
        
        # Sort teams by final skill
        team_skill_pairs = [(teams[tid], final_skills[tid]) 
                           for tid in range(len(teams))]
        team_skill_pairs.sort(key=lambda x: x[1], reverse=True)
        
        team_names = [pair[0] for pair in team_skill_pairs]
        skills = [pair[1] for pair in team_skill_pairs]
        
        fig, ax = plt.subplots(figsize=(10, max(6, len(teams) * 0.3)))
        
        # Horizontal bar chart
        y_pos = np.arange(len(team_names))
        colors = plt.cm.RdYlGn(np.linspace(0.3, 0.9, len(team_names)))
        
        bars = ax.barh(y_pos, skills, color=colors)
        
        ax.set_yticks(y_pos)
        ax.set_yticklabels(team_names, fontsize=9)
        ax.invert_yaxis()  # Highest skill at top
        ax.set_xlabel('Final Skill Estimate', fontsize=11)
        ax.set_title('Final Team Skill Rankings', fontsize=13, fontweight='bold')
        ax.axvline(x=0, color='black', linestyle='-', linewidth=0.8)
        ax.grid(True, alpha=0.3, axis='x', linestyle='--')
        
        # Add value labels on bars
        for i, (bar, skill) in enumerate(zip(bars, skills)):
            ax.text(skill, i, f' {skill:.3f}', 
                   va='center', fontsize=8, fontweight='bold')
        
        plt.tight_layout()
        plt.savefig('final_rankings.png', dpi=300, bbox_inches='tight')
        print("Final rankings saved as 'final_rankings.png'")
        plt.show()


class DataProcessor:
    """
    Handles loading and preprocessing football data.
    """
    
    @staticmethod
    def load_and_prepare_data(filepath, season_filter=None):
        """
        Load football data from CSV and prepare for filtering.
        
        Parameters:
        -----------
        filepath : str
            Path to CSV file with columns: date, home_team, away_team, result, season, result_code
        season_filter : str or list, optional
            Season(s) to include (e.g., 'PL_2018-2019' or ['PL_2018-2019', 'PL_2019-2020'])
            If None, includes all seasons
        
        Returns:
        --------
        matches : list of dict
            List of matches with keys: team_h, team_a, outcome, date
        teams : list
            List of team names (sorted)
        team_to_id : dict
            Mapping from team name to integer ID
        id_to_team : dict
            Mapping from integer ID to team name
        """
        # Load data
        print(f"Loading data from {filepath}...")
        raw_data = pd.read_csv(filepath)
        
        print(f"Total matches in file: {len(raw_data)}")
        
        # Filter by season if specified
        if season_filter is not None:
            if isinstance(season_filter, str):
                season_filter = [season_filter]
            raw_data = raw_data[raw_data['season'].isin(season_filter)]
            print(f"Matches after season filter: {len(raw_data)}")
        
        # Sort by date (chronological order is CRUCIAL!)
        raw_data['date'] = pd.to_datetime(raw_data['date'])
        raw_data = raw_data.sort_values('date').reset_index(drop=True)
        
        # Get unique teams
        home_teams = set(raw_data['home_team'])
        away_teams = set(raw_data['away_team'])
        all_teams = sorted(home_teams.union(away_teams))
        
        print(f"Number of unique teams: {len(all_teams)}")
        
        # Create team_id mappings
        team_to_id = {name: idx for idx, name in enumerate(all_teams)}
        id_to_team = {idx: name for name, idx in team_to_id.items()}
        
        # Process matches
        matches = []
        for index, row in raw_data.iterrows():
            match = {
                'team_h': team_to_id[row['home_team']],  # home team ID
                'team_a': team_to_id[row['away_team']],  # away team ID
                'date': row['date']
            }
            
            # Convert result to outcome encoding
            # result: 'H' = home win, 'A' = away win, 'D' = draw
            # outcome: 1 = home win, -1 = away win, 0 = draw
            if row['result'] == 'H':
                match['outcome'] = 1  # Home wins
            elif row['result'] == 'A':
                match['outcome'] = -1  # Away wins
            elif row['result'] == 'D':
                match['outcome'] = 0  # Draw
            else:
                print(f"Warning: Unknown result '{row['result']}' at index {index}, skipping")
                continue
            
            # Optional: store additional info
            match['home_team_name'] = row['home_team']
            match['away_team_name'] = row['away_team']
            match['season'] = row['season']
            
            matches.append(match)
        
        print(f"Processed {len(matches)} matches")
        print(f"Date range: {matches[0]['date']} to {matches[-1]['date']}")
        
        # Print outcome distribution
        outcomes = [m['outcome'] for m in matches]
        home_wins = sum(1 for o in outcomes if o == 1)
        draws = sum(1 for o in outcomes if o == 0)
        away_wins = sum(1 for o in outcomes if o == -1)
        
        print(f"\nOutcome distribution:")
        print(f"  Home wins: {home_wins} ({100*home_wins/len(matches):.1f}%)")
        print(f"  Draws:     {draws} ({100*draws/len(matches):.1f}%)")
        print(f"  Away wins: {away_wins} ({100*away_wins/len(matches):.1f}%)")
        
        return matches, all_teams, team_to_id, id_to_team
    
    @staticmethod
    def print_data_summary(matches, teams, team_to_id):
        """
        Print summary statistics about the dataset.
        
        Parameters:
        -----------
        matches : list
            List of match dictionaries
        teams : list
            List of team names
        team_to_id : dict
            Team name to ID mapping
        """
        print("\n" + "="*60)
        print("DATA SUMMARY")
        print("="*60)
        
        # Count matches per team
        team_match_counts = {tid: 0 for tid in range(len(teams))}
        for match in matches:
            team_match_counts[match['team_h']] += 1
            team_match_counts[match['team_a']] += 1
        
        print(f"\nMatches per team:")
        for team_name in teams[:5]:  # Show first 5
            tid = team_to_id[team_name]
            print(f"  {team_name}: {team_match_counts[tid]} matches")
        print("  ...")
        
        # Check for teams with very few matches
        min_matches = min(team_match_counts.values())
        max_matches = max(team_match_counts.values())
        avg_matches = np.mean(list(team_match_counts.values()))
        
        print(f"\nMatch count statistics:")
        print(f"  Min: {min_matches}")
        print(f"  Max: {max_matches}")
        print(f"  Average: {avg_matches:.1f}")
        
        if min_matches < 10:
            print(f"\n  WARNING: Some teams have very few matches!")
            for team_name in teams:
                tid = team_to_id[team_name]
                if team_match_counts[tid] < 10:
                    print(f"    {team_name}: only {team_match_counts[tid]} matches")
        
        print("="*60 + "\n")
    
    @staticmethod
    def filter_teams_by_matches(matches, teams, team_to_id, min_matches=10):
        """
        Filter out teams that don't have enough matches.
        
        Parameters:
        -----------
        matches : list
            Original list of matches
        teams : list
            Original list of teams
        team_to_id : dict
            Original team mapping
        min_matches : int
            Minimum number of matches required
        
        Returns:
        --------
        filtered_matches, filtered_teams, new_team_to_id, new_id_to_team
        """
        # Count matches per team
        team_match_counts = {tid: 0 for tid in range(len(teams))}
        for match in matches:
            team_match_counts[match['team_h']] += 1
            team_match_counts[match['team_a']] += 1
        
        # Find teams with enough matches
        valid_team_ids = {tid for tid, count in team_match_counts.items() 
                         if count >= min_matches}
        
        print(f"Teams with at least {min_matches} matches: {len(valid_team_ids)}/{len(teams)}")
        
        if len(valid_team_ids) == len(teams):
            print("All teams have sufficient matches, no filtering needed")
            id_to_team = {idx: name for name, idx in team_to_id.items()}
            return matches, teams, team_to_id, id_to_team
        
        # Filter matches
        filtered_matches = [m for m in matches 
                           if m['team_h'] in valid_team_ids and m['team_a'] in valid_team_ids]
        
        print(f"Matches after filtering: {len(filtered_matches)}/{len(matches)}")
        
        # Create new team list and mappings
        old_id_to_team = {tid: teams[tid] for tid in valid_team_ids}
        filtered_teams = sorted(old_id_to_team.values())
        new_team_to_id = {name: idx for idx, name in enumerate(filtered_teams)}
        new_id_to_team = {idx: name for name, idx in new_team_to_id.items()}
        
        # Remap team IDs in matches
        old_to_new_id = {old_tid: new_team_to_id[teams[old_tid]] 
                        for old_tid in valid_team_ids}
        
        for match in filtered_matches:
            match['team_h'] = old_to_new_id[match['team_h']]
            match['team_a'] = old_to_new_id[match['team_a']]
        
        return filtered_matches, filtered_teams, new_team_to_id, new_id_to_team


