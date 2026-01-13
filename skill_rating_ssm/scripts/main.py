import numpy as np
import pandas as pd
import os
import tqdm
from tqdm import tqdm
import pickle
from datetime import datetime

# Import classes
from smc_filtering_smoothing import (
    PairwiseSkillFilter, 
    EM_Estimator, 
    Visualizer, 
    calculate_match_log_lik
)

def compute_log_log_grid(matches, num_teams, num_particles, 
                         sigma_sq_range, epsilon_range, 
                         grid_size=20):
    """
    Computes grid points uniform in LOG-10 space.
    """
    print(f"\nComputing Log-Log Grid ({grid_size}x{grid_size})...")
    
    # 1. Convert limits to Log-10
    log_tau_min = np.log10(max(sigma_sq_range[0], 1e-6))
    log_tau_max = np.log10(sigma_sq_range[1])
    
    log_eps_min = np.log10(max(epsilon_range[0], 1e-6))
    log_eps_max = np.log10(epsilon_range[1])
    
    # 2. Create grid points in LOG space
    log_tau_vals = np.linspace(log_tau_min, log_tau_max, grid_size)
    log_eps_vals = np.linspace(log_eps_min, log_eps_max, grid_size)
    
    # 3. Convert back to Linear for the actual Filter
    tau_vals_linear = 10**log_tau_vals
    eps_vals_linear = 10**log_eps_vals
    
    log_lik_grid = np.zeros((grid_size, grid_size))
    
    # Check cache
    grid_file = "log_log_grid.pkl"
    if os.path.exists(grid_file):
        print("Loading cached Log-Log grid...")
        with open(grid_file, "rb") as f:
            return pickle.load(f)
    
    # 4. Compute Grid
    with tqdm(total=grid_size*grid_size, desc="Log-Grid") as pbar:
        for i, tau_sq in enumerate(tau_vals_linear):     # Iterate Y (Rows)
            for j, epsilon in enumerate(eps_vals_linear): # Iterate X (Cols)
                try:
                    # Run filter with linear values
                    pf = PairwiseSkillFilter(
                        num_teams=num_teams,
                        num_particles=num_particles,
                        sigma_sq=tau_sq,
                        sigma_obs_sq=1.0, # FIXED at 1.0
                        draw_threshold=epsilon
                    )
                    _, _, _, log_lik = pf.run_filter(matches)
                    log_lik_grid[i, j] = log_lik
                except Exception:
                    log_lik_grid[i, j] = -np.inf
                pbar.update(1)
                
    with open(grid_file, "wb") as f:
        # Save the LOG values for plotting
        pickle.dump((log_lik_grid, log_tau_vals, log_eps_vals), f)
        
    return log_lik_grid, log_tau_vals, log_eps_vals

def compute_fig3_grid(matches, num_teams, num_particles, 
                      sigma_sq_range, sigma_obs_sq_range, 
                      fixed_epsilon, grid_size=20):
    """
    Computes grid for Volatility (Tau) vs Observation Noise (Sigma_0).
    Matches Figure 3 of the paper.
    """
    print(f"\nComputing Figure 3 Grid ({grid_size}x{grid_size})...")
    
    # We work in LOG SPACE for the grid points to match the paper's axes
    # The ranges provided should be in linear space, we log them here
    log_tau_sq_vals = np.linspace(np.log10(sigma_sq_range[0]), np.log10(sigma_sq_range[1]), grid_size)
    log_sig_obs_sq_vals = np.linspace(np.log10(sigma_obs_sq_range[0]), np.log10(sigma_obs_sq_range[1]), grid_size)
    
    # Convert back to linear for the filter
    tau_sq_vals = 10**log_tau_sq_vals
    sig_obs_sq_vals = 10**log_sig_obs_sq_vals
    
    log_lik_grid = np.zeros((grid_size, grid_size))
    
    # Check for cached file
    grid_file = "fig3_grid_results.pkl"
    if os.path.exists(grid_file):
        print("Loading cached Figure 3 grid...")
        with open(grid_file, "rb") as f:
            return pickle.load(f)
            
    with tqdm(total=grid_size*grid_size, desc="Fig3 Grid") as pbar:
        for i, tau_sq in enumerate(tau_sq_vals):
            for j, sig_obs_sq in enumerate(sig_obs_sq_vals):
                try:
                    # Run filter with varying sigma_obs, FIXED epsilon
                    pf = PairwiseSkillFilter(
                        num_teams=num_teams,
                        num_particles=num_particles,
                        sigma_sq=tau_sq,
                        sigma_obs_sq=sig_obs_sq,
                        draw_threshold=fixed_epsilon  # <--- FIXED HERE
                    )
                    _, _, _, log_lik = pf.run_filter(matches)
                    log_lik_grid[i, j] = log_lik
                except Exception:
                    log_lik_grid[i, j] = -np.inf
                pbar.update(1)
    
    with open(grid_file, "wb") as f:
        pickle.dump((log_lik_grid, log_tau_sq_vals, log_sig_obs_sq_vals), f)
        
    return log_lik_grid, log_tau_sq_vals, log_sig_obs_sq_vals

# --- HELPER: Compute Grid for Visualization ---
def compute_log_likelihood_grid(matches, num_teams, num_particles, 
                                sigma_sq_range, epsilon_range, 
                                grid_size=10, fixed_sigma_obs_sq=1.0):
    print(f"\nComputing Grid ({grid_size}x{grid_size})...")
    
    sigma_sq_vals = np.linspace(sigma_sq_range[0], sigma_sq_range[1], grid_size)
    epsilon_vals = np.linspace(epsilon_range[0], epsilon_range[1], grid_size)
    log_lik_grid = np.zeros((grid_size, grid_size))
    
    grid_file = "grid_results.pkl"
    if os.path.exists(grid_file):
        print("Loading cached grid results...")
        with open(grid_file, "rb") as f:
            return pickle.load(f)
    
    total_evals = grid_size * grid_size
    with tqdm(total=total_evals, desc="Grid computation") as pbar:
        for i, sigma_sq in enumerate(sigma_sq_vals):
            for j, epsilon in enumerate(epsilon_vals):
                try:
                    pf = PairwiseSkillFilter(
                        num_teams=num_teams,
                        num_particles=num_particles,
                        sigma_sq=sigma_sq,
                        sigma_obs_sq=fixed_sigma_obs_sq,
                        draw_threshold=epsilon
                    )
                    _, _, _, log_lik = pf.run_filter(matches)
                    log_lik_grid[i, j] = log_lik
                except Exception as e:
                    log_lik_grid[i, j] = -np.inf
                pbar.update(1)
    
    with open(grid_file, "wb") as f:
        pickle.dump((log_lik_grid, sigma_sq_vals, epsilon_vals), f)
        
    return log_lik_grid, sigma_sq_vals, epsilon_vals

# --- MAIN EXECUTION ---
def main():
    print("\n" + "="*80)
    print(" "*20 + "FOOTBALL SKILL RATING (CUSTOM DATA)")
    print("="*80 + "\n")
    script_dir = os.path.dirname(os.path.abspath(__file__))
    # LOAD DATA 

    csv_path = os.path.join(script_dir, '../cleaned_data/cleaned_data_football.csv')
    csv_path = os.path.abspath(csv_path)
    print(f"Looking for data at: {csv_path}")
    if not os.path.exists(csv_path):
        print(f"\nError: File not found !")
        return

    print("File found! Loading...")
    df = pd.read_csv(csv_path)
    
    if not os.path.exists(csv_path):
        print(f"ERROR: File not found at {csv_path}")
        return

    print(f"Loading data from {csv_path}...")
    df = pd.read_csv(csv_path)
    
    # Filter by season
    target_season = 'PL_2018-2019'
    season_df = df[df['season'] == target_season].copy()
    
    if len(season_df) == 0:
        print(f"Error: No matches found for season '{target_season}'. Check your csv season names.")
        print(f"Available seasons: {df['season'].unique()}")
        return

    # DATE CONVERSION
    print("Converting dates...")
    season_df['date'] = pd.to_datetime(season_df['date'])
    season_df = season_df.sort_values('date')
    
    # CREATE TEAM MAPPINGS
    unique_teams = sorted(list(set(season_df['home_team'].unique()) | set(season_df['away_team'].unique())))
    team_to_id = {team: i for i, team in enumerate(unique_teams)}
    num_teams = len(unique_teams)
    
    # PREPARE MATCH LIST
    matches = []
    
    for _, row in season_df.iterrows():
        res_str = row['result'] # 'H', 'A', or 'D'
        
        if res_str == 'H':
            outcome = 1
        elif res_str == 'A':
            outcome = -1
        else:
            outcome = 0 # Draw
            
        matches.append({
            'team_h': team_to_id[row['home_team']],
            'team_a': team_to_id[row['away_team']],
            'outcome': outcome,
            'date': row['date']
        })
        
    print(f"Successfully processed {len(matches)} matches for {num_teams} teams.")

    # RUN EM ALGORITHM
    print("\nSTEP 2: Estimating Parameters (EM)...")
    em_estimator = EM_Estimator(
        matches=matches,
        num_teams=num_teams,
        num_particles=100,  
        max_iterations=50,
        tolerance=1e-3
    )
    
    em_results = em_estimator.run_EM(save_path="em_results_manual.pkl")
    
    best_sigma_sq = em_results['best_sigma_sq']
    best_epsilon = em_results['best_draw_threshold']
    
    print(f"FINAL ESTIMATES -> Volatility: {best_sigma_sq:.5f}, Draw Threshold: {best_epsilon:.5f}")

    # VISUALIZATIONS
    print("\nSTEP 3: Visualizing...")
    
    Visualizer.plot_em_convergence(em_results['em_history'])
    
    Visualizer.plot_skill_trajectories(
        skill_history=em_results['skill_history'],
        teams=unique_teams,
        team_to_id=team_to_id,
        num_teams_plot=7
    )
    
    Visualizer.plot_final_rankings(
        skill_history=em_results['skill_history'],
        teams=unique_teams
    )
    
    # GRID SEARCH
    print("\nSTEP 4: Computing Likelihood Surface...")
    sigma_range = (max(0.001, best_sigma_sq - 0.05), best_sigma_sq + 0.05)
    eps_range = (max(0.1, best_epsilon - 0.2), best_epsilon + 0.2)
    
    grid, s_vals, e_vals = compute_log_likelihood_grid(
        matches, num_teams, 100, sigma_range, eps_range, grid_size=15
    )
    
    print("\nSTEP 5: Computing Log-Log Surface...")
    
    # Define ranges (Linear Scale) covering your EM results
    # Example: Tau from 0.001 to 1.0, Epsilon from 0.1 to 5.0
    s_range = (0.001, 2.0)
    e_range = (0.1, 5.0)
    
    grid, log_tau, log_eps = compute_log_log_grid(
        matches, num_teams, 20, 
        s_range, e_range, 
        grid_size=15
    )
    
    Visualizer.plot_log_volatility_vs_log_draw(grid, log_tau, log_eps)
    # --- FIX STARTS HERE ---
    
    # Get the full history of parameter values
    hist_sigma = em_results['em_history']['sigma_sq']
    hist_eps = em_results['em_history']['draw_threshold']
    
    # Define ranges that cover the MINIMUM start point and MAXIMUM end point
    # We add 20% padding so the dots aren't on the very edge
    s_min, s_max = min(hist_sigma), max(hist_sigma)
    e_min, e_max = min(hist_eps), max(hist_eps)
    
    # Create a grid that covers the ENTIRE journey
    sigma_range = (s_min * 0.8, s_max * 1.2)
    eps_range = (e_min * 0.8, e_max * 1.2)
    
    print(f"Grid Range sigma: {sigma_range}")
    print(f"Grid Range epsilon: {eps_range}")

    # Now compute the grid with these wider ranges
    grid, s_vals, e_vals = compute_log_likelihood_grid(
        matches, num_teams, 
        num_particles=100, 
        sigma_sq_range=sigma_range, 
        epsilon_range=eps_range, 
        grid_size=15  # 15x15 is good for speed
    )
    Visualizer.create_log_lik_surface(
        grid, s_vals, e_vals, best_sigma_sq, best_epsilon, em_results['em_history']
    )
    
    # ... inside main() ...

    # 8. FIGURE 3 REPLICA (The "Ridge" Plot)
    print("\nSTEP 5: Generating Figure 3 Replica (Log-Log Scale)...")
    
    # We need to define wide LOG ranges to see the shape
    # Volatility range: 10^-4 to 10^0
    tau_range = (0.0001, 1.0) 
    # Obs Noise range: 10^-1 to 10^1
    obs_range = (0.1, 10.0)
    
    # Use the best epsilon we found earlier
    fixed_eps = best_epsilon 
    
    grid_fig3, log_tau, log_obs = compute_fig3_grid(
        matches, num_teams, 100, 
        tau_range, obs_range, 
        fixed_epsilon=fixed_eps, 
        grid_size=15
    )
    
    Visualizer.plot_figure_3_replica(grid_fig3, log_tau, log_obs)
    print("\nAnalysis Complete! Check .png files.")

if __name__ == "__main__":
    main()