import numpy as np
from smc_filtering_smoothing import (
    SkillRatingModel, 
    PairwiseSkillFilter, 
    EM_Estimator, 
    Visualizer, 
    DataProcessor
)

def compute_log_likelihood_grid(matches, num_teams, num_particles, 
                                sigma_sq_range, sigma_obs_sq_range, 
                                grid_size=20, draw_threshold=0.5):
    """
    Compute log-likelihood over a grid of parameter values for Figure 3.
    
    Parameters:
    -----------
    matches : list
        List of matches
    num_teams : int
        Number of teams
    num_particles : int
        Number of particles for filter
    sigma_sq_range : tuple
        (min, max) for σ²
    sigma_obs_sq_range : tuple
        (min, max) for sigma_squared_obs
    grid_size : int
        Number of points in each dimension
    draw_threshold : float
        Threshold for draws
    
    Returns:
    --------
    log_lik_grid : 2D array
        Grid of log-likelihoods
    sigma_sq_values : 1D array
        σ² values
    sigma_obs_sq_values : 1D array
        sigma_squared_obs values
    """
    print("\n" + "="*60)
    print("COMPUTING LOG-LIKELIHOOD GRID")
    print("="*60)
    print(f"Grid size: {grid_size} x {grid_size} = {grid_size**2} evaluations")
    print(f"This may take a while...")
    
    # Create grid
    sigma_sq_values = np.linspace(sigma_sq_range[0], sigma_sq_range[1], grid_size)
    sigma_obs_sq_values = np.linspace(sigma_obs_sq_range[0], sigma_obs_sq_range[1], grid_size)
    
    log_lik_grid = np.zeros((grid_size, grid_size))
    
    total_evaluations = grid_size * grid_size
    completed = 0
    
    for i, sigma_sq in enumerate(sigma_sq_values):
        for j, sigma_obs_sq in enumerate(sigma_obs_sq_values):
            completed += 1
            if completed % 10 == 0 or completed == total_evaluations:
                print(f"Progress: {completed}/{total_evaluations} ({100*completed/total_evaluations:.1f}%)")
            
            # Run particle filter
            try:
                pf = PairwiseSkillFilter(
                    num_teams=num_teams,
                    num_particles=num_particles,
                    sigma_sq=sigma_sq,
                    sigma_obs_sq=sigma_obs_sq,
                    draw_threshold=draw_threshold
                )
                _, _, _, log_lik = pf.run_filter(matches)
                log_lik_grid[i, j] = log_lik
            except Exception as e:
                print(f"  Error at grid point ({i},{j}): {e}")
                log_lik_grid[i, j] = -np.inf
    
    print("Grid computation completed!")
    print("="*60 + "\n")
    
    return log_lik_grid, sigma_sq_values, sigma_obs_sq_values


def main():
    """
    Main execution function for skill rating analysis.
    """
    print("\n" + "="*80)
    print(" "*20 + "FOOTBALL SKILL RATING ANALYSIS")
    print("="*80 + "\n")
    
    # ========== STEP 1: Load and prepare data ==========
    print("STEP 1: Loading and preparing data")
    print("-"*80)
    
    # Load data for a specific season (or multiple seasons)
    matches, teams, team_to_id, id_to_team = DataProcessor.load_and_prepare_data(
        filepath = '/Users/wissalhaouami/projects/skill_rating_ssm/code/notebooks/scripts/cleaned_football_data.csv',
        season_filter = None
    )
    
    # Print data summary
    DataProcessor.print_data_summary(matches, teams, team_to_id)
    
    num_teams = len(teams)
    num_matches = len(matches)
    
    print(f"\nDataset summary:")
    print(f"  Teams: {num_teams}")
    print(f"  Matches: {num_matches}")
    print(f"  Average matches per team: {2*num_matches/num_teams:.1f}")
    
    # ========== STEP 2: Run EM algorithm ==========
    print("\n" + "="*80)
    print("STEP 2: Running EM algorithm")
    print("-"*80)
    
    em_estimator = EM_Estimator(
        matches=matches,
        num_teams = num_teams,
        num_particles = 1000,  # Increase for better accuracy, decrease for speed
        max_iterations=50,
        tolerance=1e-4
    )
    
    em_sigma_sq, em_sigma_obs_sq, param_history, log_lik_history = em_estimator.run_EM()
    
    print("\n" + "="*80)
    print("EM RESULTS")
    print("="*80)
    print(f"Final parameter estimates:")
    print(f"  σ² (skill evolution variance):    {em_sigma_sq:.6f}")
    print(f"  σ²_obs (observation noise):       {em_sigma_obs_sq:.6f}")
    print(f"  Final log-likelihood:             {log_lik_history[-1]:.4f}")
    print(f"  Number of EM iterations:          {len(param_history)}")
    print("="*80 + "\n")
    
    # ========== STEP 3: Plot EM convergence ==========
    print("STEP 3: Plotting EM convergence")
    print("-"*80)
    
    Visualizer.plot_em_convergence(log_lik_history, param_history)
    
    # ========== STEP 4: Compute log-likelihood grid ==========
    print("\nSTEP 4: Computing log-likelihood grid for Figure 3")
    print("-"*80)
    
    # Define grid ranges based on EM estimates
    # Use ±50% around EM estimates, or fixed ranges
    sigma_sq_min = max(0.01, em_sigma_sq * 0.5)
    sigma_sq_max = em_sigma_sq * 1.5
    sigma_obs_sq_min = max(0.1, em_sigma_obs_sq * 0.5)
    sigma_obs_sq_max = em_sigma_obs_sq * 1.5
    
    print(f"Grid ranges:")
    print(f"  σ² range:     [{sigma_sq_min:.4f}, {sigma_sq_max:.4f}]")
    print(f"  σ²_obs range: [{sigma_obs_sq_min:.4f}, {sigma_obs_sq_max:.4f}]")
    
    log_lik_grid, sigma_sq_vals, sigma_obs_sq_vals = compute_log_likelihood_grid(
        matches=matches,
        num_teams=num_teams,
        num_particles=500,  # Use fewer particles for speed
        sigma_sq_range=(sigma_sq_min, sigma_sq_max),
        sigma_obs_sq_range=(sigma_obs_sq_min, sigma_obs_sq_max),
        grid_size=20,  # Start with 20x20, increase to 30x30 for final version
        draw_threshold=0.5
    )
    
    # ========== STEP 5: Create Figure 3 ==========
    print("STEP 5: Creating Figure 3 (log-likelihood surface)")
    print("-"*80)
    
    Visualizer.create_figure_3(
        log_lik_grid=log_lik_grid,
        sigma_sq_vals=sigma_sq_vals,
        sigma_obs_sq_vals=sigma_obs_sq_vals,
        em_sigma_sq=em_sigma_sq,
        em_sigma_obs_sq=em_sigma_obs_sq,
        param_history=param_history
    )
    
    # ========== STEP 6: Run final filter and analyze skills ==========
    print("\nSTEP 6: Running final filter with estimated parameters")
    print("-"*80)
    
    final_filter = PairwiseSkillFilter(
        num_teams=num_teams,
        num_particles=1000,
        sigma_sq=em_sigma_sq,
        sigma_obs_sq=em_sigma_obs_sq,
        draw_threshold=0.5
    )
    
    print("Running filter...")
    particles, weights, skill_history, final_log_lik = final_filter.run_filter(matches)
    
    print(f"Final log-likelihood: {final_log_lik:.4f}")
    
    # ========== STEP 7: Visualize skill trajectories ==========
    print("\nSTEP 7: Visualizing skill trajectories")
    print("-"*80)
    
    # Plot top 10 teams
    Visualizer.plot_skill_trajectories(
        skill_history=skill_history,
        teams=teams,
        team_to_id=team_to_id,
        num_teams_plot=10
    )
    
    # Plot final rankings
    Visualizer.plot_final_skill_ranking(
        skill_history=skill_history,
        teams=teams
    )
    
    # ========== STEP 8: Print final rankings ==========
    print("\nSTEP 8: Final team rankings")
    print("="*80)
    
    final_skills = skill_history[-1]
    rankings = [(id_to_team[tid], final_skills[tid]) for tid in range(num_teams)]
    rankings.sort(key=lambda x: x[1], reverse=True)
    
    print(f"\n{'Rank':<6}{'Team':<25}{'Skill':<12}")
    print("-"*80)
    for rank, (team, skill) in enumerate(rankings, 1):
        print(f"{rank:<6}{team:<25}{skill:>10.4f}")
    
    print("\n" + "="*80)
    print("ANALYSIS COMPLETE!")
    print("="*80)
    print("\nGenerated files:")
    print("  - figure_3_football.png         : Log-likelihood surface")
    print("  - em_convergence.png            : EM algorithm convergence")
    print("  - skill_trajectories.png        : Top teams' skill evolution")
    print("  - final_rankings.png            : Final skill rankings")
    print("\n" + "="*80 + "\n")


if __name__ == "__main__":
    main()