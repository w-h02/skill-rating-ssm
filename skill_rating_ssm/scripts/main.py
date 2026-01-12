import numpy as np
import pandas as pd
import os
import tqdm
from tqdm import tqdm
import pickle
from datetime import datetime

# Import your classes (Ensure these are in your smc_filtering_smoothing.py)
from smc_filtering_smoothing import (
    PairwiseSkillFilter, 
    EM_Estimator, 
    Visualizer, 
    calculate_match_log_lik
)

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
    # 1. LOAD DATA 
    # Use the absolute path if relative paths are failing
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

    # 2. DATE CONVERSION
    print("Converting dates...")
    # Your date format is YYYY-MM-DD, which pandas handles automatically usually
    season_df['date'] = pd.to_datetime(season_df['date'])
    season_df = season_df.sort_values('date')
    
    # 3. CREATE TEAM MAPPINGS
    unique_teams = sorted(list(set(season_df['home_team'].unique()) | set(season_df['away_team'].unique())))
    team_to_id = {team: i for i, team in enumerate(unique_teams)}
    num_teams = len(unique_teams)
    
    # 4. PREPARE MATCH LIST
    # Mapping 'result' (H, D, A) to numbers (1, 0, -1)
    # We ignore 'result_code' from CSV to ensure standard logic
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

    # 5. RUN EM ALGORITHM
    print("\nSTEP 2: Estimating Parameters (EM)...")
    em_estimator = EM_Estimator(
        matches=matches,
        num_teams=num_teams,
        num_particles=50,  
        max_iterations=30,
        tolerance=1e-3
    )
    
    em_results = em_estimator.run_EM(save_path="em_results_manual.pkl")
    
    best_sigma_sq = em_results['best_sigma_sq']
    best_epsilon = em_results['best_draw_threshold']
    
    print(f"FINAL ESTIMATES -> Volatility: {best_sigma_sq:.5f}, Draw Threshold: {best_epsilon:.5f}")

    # 6. VISUALIZATIONS
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
    
    # 7. GRID SEARCH
    print("\nSTEP 4: Computing Likelihood Surface...")
    sigma_range = (max(0.001, best_sigma_sq - 0.05), best_sigma_sq + 0.05)
    eps_range = (max(0.1, best_epsilon - 0.2), best_epsilon + 0.2)
    
    grid, s_vals, e_vals = compute_log_likelihood_grid(
        matches, num_teams, 200, sigma_range, eps_range, grid_size=15
    )
    
    Visualizer.create_log_lik_surface(
        grid, s_vals, e_vals, best_sigma_sq, best_epsilon, em_results['em_history']
    )
    
    print("\nAnalysis Complete! Check .png files.")

if __name__ == "__main__":
    main()