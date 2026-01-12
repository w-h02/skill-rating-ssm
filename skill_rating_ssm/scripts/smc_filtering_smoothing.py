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
import pickle
import os

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
    
        self.particles = {}  # particles per team
        self.weights = {}    # weights per team

        for team_id in range(num_teams):
            model = SkillRatingModel(sigma_sq=self.sigma_sq)
            # Initial distribution and weights
            initial_dist = model.PX0()
            self.particles[team_id] = initial_dist.rvs(size=self.num_particles)
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
        previous_date = matches[0]['date']
        
        for match in tqdm(matches, desc="Processing matches", leave=False):
            team_home = match['team_h']
            team_away = match['team_a']
            outcome = match['outcome']
            current_date = match['date']

            delta_t = (current_date - previous_date).days
            delta_t = max(0, delta_t)
            # PREDICT STEP: propagate all teams forward
            if delta_t > 0:
                self.predict_all_teams(delta_t)
            
            previous_date = current_date

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
    
    def predict_all_teams(self, delta_t = 1):
        for team_id in range(self.num_teams):
            noise = np.random.normal(
                loc=0.0,
                scale=np.sqrt(self.sigma_sq * delta_t),
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
        Compute P(outcome | skill_diff) 
        """
        # Denominator s = sqrt(2 * sigma_obs^2)
        scale = np.sqrt(2 * self.sigma_obs_sq)
    
        if outcome == 1:  # Home Win
            #Performance must be GREATER than the draw threshold
            # P(d + noise > epsilon)
            return max(norm.cdf((skill_diff - self.draw_threshold) / scale), 1e-10)
        
        elif outcome == -1:  # Away Win
            # Performance must be LESS than negative draw threshold
            # P(d + noise < -epsilon)
            return max(norm.cdf((-skill_diff - self.draw_threshold) / scale), 1e-10)
        
        else:  # Draw
            # Performance is BETWEEN -epsilon and epsilon
            z_upper = (self.draw_threshold - skill_diff) / scale
            z_lower = (-self.draw_threshold - skill_diff) / scale
            return max(norm.cdf(z_upper) - norm.cdf(z_lower), 1e-10)
    
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
    
# Calculate match log likelihood

def calculate_match_log_lik(skill_diff, outcome, sigma_obs_sq, draw_threshold):
    """
    Computes log P(y | diff, parameters) for the Ordered Probit model.
    Used by both the Filter (for weighting) and the EM (for optimization).
    """
    scale = np.sqrt(2 * sigma_obs_sq)
    epsilon = draw_threshold
    
    if outcome == 1:  # Home Win
        # P(diff + noise > epsilon)
        p = norm.cdf((skill_diff - epsilon) / scale)
        
    elif outcome == -1:  # Away Win
        # P(diff + noise < -epsilon)
        p = norm.cdf((-skill_diff - epsilon) / scale)
        
    else:  # Draw
        # P(-epsilon < diff + noise < epsilon)
        z_upper = (epsilon - skill_diff) / scale
        z_lower = (-epsilon - skill_diff) / scale
        p = norm.cdf(z_upper) - norm.cdf(z_lower)
        
    return np.log(max(p, 1e-10)) # Avoid log(0)

# EM Algorithm

class EM_Estimator:
    def __init__(self, matches, num_teams, num_particles=1000, max_iterations=30, tolerance=1e-4):
        self.matches = matches
        self.num_teams = num_teams
        self.num_particles = num_particles
        self.max_iterations = max_iterations
        self.tolerance = tolerance
        
        # FIXED parameter 
        self.fixed_sigma_obs_sq = 1.0 

    def run_EM(self, save_path="em_results.pkl"):
        # Check if results already exist
        if os.path.exists(save_path):
            print(f"Loading cached results from {save_path}...")
            with open(save_path, "rb") as f:
                return pickle.load(f)

        sigma_sq = 0.1         # Initial guess for volatility
        draw_threshold = 0.3   # Initial guess for draw width
        
        history_log = {'sigma_sq': [], 'draw_threshold': [], 'log_lik': []}

        print(f"Starting EM... Fixing sigma_obs_sq={self.fixed_sigma_obs_sq}")
        
        for it in range(self.max_iterations):
            print(f"--- Iteration {it+1}/{self.max_iterations} ---")
            
            # === E-STEP: Run Filter with current params ===
            pf = PairwiseSkillFilter(
                self.num_teams, self.num_particles, 
                sigma_sq, self.fixed_sigma_obs_sq, draw_threshold
            )
            _, _, skill_history, log_lik = pf.run_filter(self.matches)
            
            history_log['log_lik'].append(log_lik)
            history_log['sigma_sq'].append(sigma_sq)
            history_log['draw_threshold'].append(draw_threshold)
            print(f"Log-Likelihood: {log_lik:.2f} | sigma_sq: {sigma_sq:.4f} | epsilon: {draw_threshold:.4f}")

            # === M-STEP: Update Parameters ===
            
            # Update Sigma Squared (Volatility)
            # Formula: Mean squared displacement / delta_t

            sq_diffs = 0
            count = 0
            for t in range(1, len(skill_history)):
                for team in range(self.num_teams):
                    diff = skill_history[t][team] - skill_history[t-1][team]
                    sq_diffs += diff ** 2
                    count += 1
            
            new_sigma_sq = sq_diffs / count
            
            # Update Draw Threshold (Epsilon)
            # We maximize the likelihood of outcomes given the fixed skills from E-step
            res = optimize.minimize_scalar(
                lambda eps: -self._objective_epsilon(eps, skill_history),
                bounds=(0.01, 2.0),
                method='bounded'
            )
            new_draw_threshold = res.x

            # Convergence Check
            if (abs(new_sigma_sq - sigma_sq) < self.tolerance and 
                abs(new_draw_threshold - draw_threshold) < self.tolerance):
                print("Converged!")
                sigma_sq, draw_threshold = new_sigma_sq, new_draw_threshold
                break
                
            sigma_sq = new_sigma_sq
            draw_threshold = new_draw_threshold

        # === FINAL RUN ===
        final_pf = PairwiseSkillFilter(
            self.num_teams, self.num_particles, 
            sigma_sq, self.fixed_sigma_obs_sq, draw_threshold
        )
        particles, weights, final_history, final_lik = final_pf.run_filter(self.matches)
        
        results = {
            "best_sigma_sq": sigma_sq,
            "best_draw_threshold": draw_threshold,
            "skill_history": final_history,
            "em_history": history_log,
            "final_particles": particles
        }
        
        # Save to disk
        with open(save_path, "wb") as f:
            pickle.dump(results, f)
            
        return results

    def _objective_epsilon(self, epsilon, skill_history):
        """Helper to optimize epsilon given fixed skills"""
        total_log_lik = 0
        for i, match in enumerate(self.matches):
            sh = skill_history[i][match['team_h']]
            sa = skill_history[i][match['team_a']]
            
            # Use the shared helper!
            lik = calculate_match_log_lik(
                sh - sa, match['outcome'], self.fixed_sigma_obs_sq, epsilon
            )
            total_log_lik += lik
        return total_log_lik


class Visualizer:
    """
    Creates plots and visualizations for the Particle Filter / EM project.
    """
    
    @staticmethod
    def create_log_lik_surface(log_lik_grid, sigma_sq_vals, epsilon_vals,
                               best_sigma_sq, best_epsilon, em_history=None):
        """
        Create the log-likelihood surface plot (Matches Slide 26).
        
        Parameters:
        -----------
        log_lik_grid : 2D array
            Grid of log-likelihoods (grid_size x grid_size)
        sigma_sq_vals : 1D array
            Values for Dynamics parameter τ² (y-axis)
        epsilon_vals : 1D array
            Values for Draw parameter ε (x-axis)
        best_sigma_sq : float
            Final EM estimate of τ²
        best_epsilon : float
            Final EM estimate of ε
        em_history : dict
            Dictionary with 'sigma_sq' and 'draw_threshold' lists
        """
        fig, ax = plt.subplots(figsize=(10, 8))
        
        # Create meshgrid for contour plot
        X, Y = np.meshgrid(epsilon_vals, sigma_sq_vals)
        
        contour = ax.contourf(X, Y, log_lik_grid, levels=25, cmap='viridis')
        cbar = plt.colorbar(contour, ax=ax)
        cbar.set_label('Log-Likelihood', rotation=270, labelpad=20)
        
        ax.contour(X, Y, log_lik_grid, levels=10, colors='white', 
                  alpha=0.3, linewidths=0.5)
        
        # Final Estimate
        ax.plot(best_epsilon, best_sigma_sq, 'r*', markersize=20, 
               label='Final Estimate', markeredgecolor='white', markeredgewidth=1.5)
        
        # Plot EM convergence path
        if em_history is not None:
            path_sigma = em_history['sigma_sq']
            path_eps = em_history['draw_threshold']
            
            ax.plot(path_eps, path_sigma, 'w--', 
                   linewidth=2, label='EM Path', alpha=0.8)
            
            # Mark starting point
            ax.plot(path_eps[0], path_sigma[0], 'go', 
                   markersize=10, label='Start', markeredgecolor='white')
            
            if len(path_eps) > 1:
                ax.annotate('', 
                           xy=(path_eps[1], path_sigma[1]),
                           xytext=(path_eps[0], path_sigma[0]),
                           arrowprops=dict(arrowstyle='->', color='white', lw=1.5))
        
        ax.set_xlabel(r'Draw Parameter $\epsilon$', fontsize=12)
        ax.set_ylabel(r'Dynamics Parameter $\tau^2$ (Volatility)', fontsize=12)
        ax.set_title('Log-Likelihood Surface (Grid Search vs EM Path)', 
                    fontsize=14, fontweight='bold')
        ax.legend(loc='best', fontsize=10, frameon=True, facecolor='white', framealpha=0.9)
        ax.grid(True, alpha=0.3, linestyle='--')
        
        plt.tight_layout()
        plt.savefig('log_lik_surface.png', dpi=300)
        print("Log-likelihood surface saved as 'log_lik_surface.png'")
        plt.show()

    @staticmethod
    def plot_skill_trajectories(skill_history, teams, team_to_id, 
                                teams_to_plot=None, num_teams_plot=7):
        """
        Plot skill evolution over time (Matches Slide 24).
        """
        if teams_to_plot is None:
            # Select top teams based on final skill
            final_skills = skill_history[-1]
            top_team_ids = sorted(final_skills.keys(), 
                                 key=lambda x: final_skills[x], 
                                 reverse=True)[:num_teams_plot]
            teams_to_plot = [teams[tid] for tid in top_team_ids]
        
        fig, ax = plt.subplots(figsize=(12, 6))

        colors = plt.cm.tab10(np.linspace(0, 1, len(teams_to_plot)))
        
        for idx, team_name in enumerate(teams_to_plot):
            team_id = team_to_id[team_name]
            trajectory = [skill_history[t][team_id] for t in range(len(skill_history))]
            
            ax.plot(trajectory, label=team_name, linewidth=2, color=colors[idx], alpha=0.8)
        
        ax.set_xlabel('Match Index $k$', fontsize=12)
        ax.set_ylabel('Posterior Mean of Skill', fontsize=12)
        ax.set_title('SMC: Posterior Mean Trajectories (Selected Teams)', fontsize=14, fontweight='bold')
        ax.legend(bbox_to_anchor=(1.02, 1), loc='upper left', fontsize=10)
        ax.grid(True, alpha=0.3, linestyle='--')
        ax.axhline(y=0, color='black', linestyle='-', linewidth=0.8, alpha=0.3)
        
        plt.tight_layout()
        plt.savefig('skill_trajectories.png', dpi=300)
        print("Skill trajectories saved as 'skill_trajectories.png'")
        plt.show()
    
    @staticmethod
    def plot_em_convergence(em_history):
        """
        Plot EM convergence (Matches Slide 23).
        
        Parameters:
        -----------
        em_history : dict
            Dictionary containing lists for 'log_lik', 'sigma_sq', 'draw_threshold'
        """
        log_liks = em_history['log_lik']
        sigma_sqs = em_history['sigma_sq']
        epsilons = em_history['draw_threshold']
        iterations = range(len(log_liks))
        
        fig, axes = plt.subplots(2, 1, figsize=(10, 8), sharex=True)
        
        # Log-Likelihood Plot
        axes[0].plot(iterations, log_liks, 'b-o', linewidth=2, markersize=5)
        axes[0].set_ylabel('Log-Likelihood', fontsize=11)
        axes[0].set_title('EM Convergence: Log-Likelihood', fontsize=12, fontweight='bold')
        axes[0].grid(True, alpha=0.3)
        
        # Parameters Plot 
        ax2 = axes[1]
        ax2.plot(iterations, sigma_sqs, 'r-o', linewidth=2, markersize=5, 
                 label=r'$\tau^2$ (Skill Evolution)')
        ax2.set_ylabel(r'$\tau^2$', fontsize=12, color='r')
        ax2.tick_params(axis='y', labelcolor='r')
        ax2.grid(True, alpha=0.3)
        
        # Second y-axis for Epsilon
        ax3 = ax2.twinx()
        ax3.plot(iterations, epsilons, 'g-s', linewidth=2, markersize=5, 
                 label=r'$\epsilon$ (Draw Threshold)')
        ax3.set_ylabel(r'$\epsilon$', fontsize=12, color='g')
        ax3.tick_params(axis='y', labelcolor='g')
        ax3.set_xlabel('EM Iteration', fontsize=12)
        
        # Combined Legend
        lines1, labels1 = ax2.get_legend_handles_labels()
        lines2, labels2 = ax3.get_legend_handles_labels()
        ax2.legend(lines1 + lines2, labels1 + labels2, loc='center right')
        
        axes[1].set_title('EM Convergence: Parameters', fontsize=12, fontweight='bold')
        
        plt.tight_layout()
        plt.savefig('em_convergence.png', dpi=300)
        print("EM convergence plot saved as 'em_convergence.png'")
        plt.show()

    @staticmethod
    def plot_final_rankings(skill_history, teams):
        """
        Plot horizontal bar chart of final rankings (Matches Slide 25).
        """
        final_skills = skill_history[-1]
        
        # Create (Team, Skill) tuples and sort
        ranking = []
        for i, team_name in enumerate(teams):
            ranking.append((team_name, final_skills[i]))
        
        # Sort descending
        ranking.sort(key=lambda x: x[1], reverse=False) 
        
        names = [x[0] for x in ranking]
        values = [x[1] for x in ranking]
        
        fig, ax = plt.subplots(figsize=(8, len(teams) * 0.4))
        
        norm = plt.Normalize(min(values), max(values))
        colors = plt.cm.RdYlGn(norm(values))
        
        bars = ax.barh(names, values, color=colors, edgecolor='grey', alpha=0.9)
        
        ax.set_xlabel('Final Skill Estimate')
        ax.set_title(f'Final Team Rankings (Season End)', fontweight='bold')
        ax.axvline(0, color='black', linewidth=0.8)
        ax.grid(axis='x', alpha=0.3, linestyle='--')

        for bar, val in zip(bars, values):
            width = bar.get_width()
            label_x_pos = width + (0.1 if width >= 0 else -0.1)
            ha = 'left' if width >= 0 else 'right'
            ax.text(label_x_pos, bar.get_y() + bar.get_height()/2, 
                    f'{val:.2f}', va='center', ha=ha, fontsize=8)
            
        plt.tight_layout()
        plt.savefig('final_rankings.png', dpi=300)
        print("Final rankings saved as 'final_rankings.png'")
        plt.show()