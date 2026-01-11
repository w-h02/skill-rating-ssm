# Football Skill Rating with Particle Filters and EM

This project implements a **dynamic skill rating system** for football teams using **state-space models**, **particle filters (SMC)**, and an **EM algorithm** to estimate model parameters. It tracks skill evolution over time, estimates match outcome probabilities, and generates visualizations of team skills and EM convergence.

---

## Table of Contents
- [Overview](#overview)  
- [Features](#features)  
- [Installation](#installation)  
- [Data Requirements](#data-requirements)  
- [Project Structure](#project-structure)  
- [Usage](#usage)  
- [Configuration](#configuration)  
- [Model Details](#model-details)  
- [Output & Visualizations](#output--visualizations)  
- [Performance Considerations](#performance-considerations)  
- [Contributing](#contributing)  
- [License](#license)  
- [References](#references)

---

## Overview

This project estimates football team skills using a **Bayesian state-space model**:
- Teams' latent skills evolve over time as a **random walk**.
- Match outcomes are modeled using a **Thurstone-Mosteller likelihood**:
  - Win/loss/draw probabilities based on skill differences.
- **Particle filters (SMC)** are used for approximate inference.
- **EM algorithm** estimates skill evolution variance (`σ²`) and observation noise (`σ²_obs`).

The system provides robust skill estimates even with limited data and handles the non-linear dynamics of competitive sports.

---

## Features

✅ **Data Processing**  
- Load and preprocess football match data (CSV format)  
- Filter teams with insufficient matches  
- Handle multiple seasons of data (Premier League 2018-2025)

✅ **Advanced Statistical Methods**  
- **Expectation-Maximization (EM)** algorithm for parameter estimation  
- **Sequential Monte Carlo (Particle Filtering)** for state estimation  
- Automatic convergence detection with tolerance thresholds

✅ **Parameter Estimation**  
- Estimate `σ²` (skill evolution variance)  
- Estimate `σ²_obs` (observation noise)  
- Track parameter history and log-likelihood evolution

✅ **Visualizations**  
- Skill trajectories over time for all teams  
- Final skill rankings with uncertainty estimates  
- EM algorithm convergence plots  
- Log-likelihood surfaces for parameter exploration

---

## Installation

### 1. Clone the repository
```bash
git clone https://github.com/w-h002/skill_rating_ssm.git
cd skill_rating_ssm
```

### 2. Create a virtual environment (recommended)
```bash
python -m venv .venv
source .venv/bin/activate  # On Windows: .venv\Scripts\activate
```

### 3. Install dependencies
```bash
pip install -r requirements.txt
```

---

## Data Requirements

### Input Format
The system expects CSV files with match data containing:
- **Team identifiers** (home team, away team)
- **Match outcomes** (win/loss/draw or scores)
- **Temporal ordering** (date or season information)

### Example Data Structure
```
Date,Home_Team,Away_Team,Home_Score,Away_Score
2023-08-12,Arsenal,Manchester City,2,1
2023-08-13,Liverpool,Chelsea,1,1
...
```

### Datasets Included
The repository includes Premier League data from:
- `PL_2018-2019.csv`
- `PL_2019-2020.csv`
- `PL_2020-2021.csv`
- `PL_2021-2022.csv`
- `PL_2022-2023.csv`
- `PL_2023-2024.csv`
- `PL_2024-2025.csv`

---

## Project Structure

```
skill_rating_ssm/
├── cleaned_data/              # Processed datasets
│   ├── cleaned_data_football.csv
│   └── cleaned_football_data.csv
├── data_prep/                 # Data preparation scripts and raw data
│   ├── chess datasets/
│   ├── football datasets/
│   │   ├── PL_2018-2019.csv
│   │   ├── PL_2019-2020.csv
│   │   ├── PL_2020-2021.csv
│   │   ├── PL_2021-2022.csv
│   │   ├── PL_2022-2023.csv
│   │   ├── PL_2023-2024.csv
│   │   └── PL_2024-2025.csv
│   └── 01_data_preparation.ipynb
├── notebooks/                 # Analysis notebooks
│   ├── parameter_estimation.ipynb
│   └── smc_filtering.ipynb
├── plots/                     # Generated visualizations
│   ├── likelihood_surface.png
│   └── skill_trajectories.png
├── scripts/                   # Main execution scripts
│   ├── main.py
│   └── smc_filtering_smoothing.py
├── README.md
└── requirements.txt
```

---

## Usage

### Running the Main Script

Execute the complete skill rating pipeline:

```bash
python scripts/main.py
```

**This will:**
1. Load cleaned match data
2. Initialize parameters with reasonable defaults
3. Run the EM algorithm for parameter estimation
4. Perform particle filtering for skill estimation
5. Output final parameter estimates and log-likelihood
6. Generate visualizations (if enabled)

### Using Jupyter Notebooks

#### Data Preparation
```bash
jupyter notebook data_prep/01_data_preparation.ipynb
```

#### Parameter Estimation Analysis
```bash
jupyter notebook notebooks/parameter_estimation.ipynb
```

#### SMC Filtering Exploration
```bash
jupyter notebook notebooks/smc_filtering.ipynb
```

---

## Configuration

### Key Parameters in `main.py`

```python
# Particle Filter Configuration
num_particles = 10        # Number of particles (increase for better accuracy)

# EM Algorithm Configuration
max_iterations = 50       # Maximum number of EM iterations
tolerance = 1e-4          # Convergence threshold for parameter changes

# Initial Parameter Guesses
initial_sigma_sq = 10.0   # Initial skill evolution variance
initial_sigma_obs_sq = 1.6 # Initial observation noise
```

### Performance Tuning

| Parameter | Effect | Recommended Range |
|-----------|--------|-------------------|
| `num_particles` | Accuracy vs. Speed | 10-100 |
| `max_iterations` | Convergence quality | 30-100 |
| `tolerance` | Early stopping | 1e-3 to 1e-5 |

---

## Model Details

### State-Space Model

**State Equation (Skill Evolution):**
```
s_t = s_{t-1} + w_t,  w_t ~ N(0, σ²)
```
- Skills evolve as a random walk
- `σ²` controls how much skills change between matches

**Observation Equation (Match Outcomes):**
```
y_t ~ Thurstone-Mosteller(s_home - s_away, σ²_obs)
```
- Match outcomes depend on skill difference
- `σ²_obs` represents match uncertainty/observation noise

### EM Algorithm

The algorithm alternates between:

1. **E-Step (Expectation):**
   - Run particle filter with current parameter estimates
   - Compute expected sufficient statistics

2. **M-Step (Maximization):**
   - Update `σ²` and `σ²_obs` to maximize expected log-likelihood
   - Use closed-form updates based on sufficient statistics

**Convergence Criteria:**
```
|σ²_new - σ²_old| < tolerance AND |σ²_obs_new - σ²_obs_old| < tolerance
```

### Particle Filter (SMC)

- **Initialization:** Sample initial skills from prior distribution
- **Prediction:** Evolve particles according to state equation
- **Update:** Weight particles by match outcome likelihood
- **Resampling:** Systematic resampling to avoid degeneracy
- **Output:** Weighted average of particles for skill estimates

---

## Output & Visualizations

### Console Output

```
Running EM algorithm...
================================================================================
EM Iteration: 34/50

Running particle filter...
og-likelihood: -3464.6986
Current parameters: sigma_sq = 10.000000, sigma_obs_sq = 1.673436
Updating parameters...
New parameters: sigma_sq = 10.000000, sigma_obs_sq = 1.349016
Parameter change: 0.32442067 (tolerance: 0.0001)
================================================================================
EM RESULTS
Final parameter estimates:
  σ² (skill evolution variance):    10.00
  σ²_obs (observation noise):       1.35
```

### Generated Plots

1. **Skill Trajectories** (`plots/skill_trajectories.png`)
   - Time series of each team's skill evolution
   - Shows skill dynamics throughout the season

2. **Final Skill Rankings** (displayed in output)
   - Bar chart of final team skills with uncertainty
   - Ranked from strongest to weakest

3. **Likelihood Surface** (`plots/likelihood_surface.png`)
   - 2D heatmap of log-likelihood over parameter space
   - Shows EM convergence path

4. **EM Convergence** (displayed in output)
   - Parameter evolution across iterations
   - Log-likelihood improvement over time

---

## Performance Considerations

### Computational Complexity

- **Time complexity:** O(T × N × P)
  - T = number of matches
  - N = number of teams
  - P = number of particles

- **Memory complexity:** O(N × P)

### Optimization Tips

1. **Increase `num_particles`** for better accuracy (10 → 50 → 100)
2. **Start with small datasets** to test parameters
3. **Monitor convergence** - algorithm typically converges in 20-40 iterations
4. **Use profiling** to identify bottlenecks for large datasets

### Typical Runtime

- Small dataset (500 matches, 20 teams, 10 particles): ~1-2 minutes
- Medium dataset (2000 matches, 20 teams, 50 particles): ~5-10 minutes
- Large dataset (5000 matches, 50 teams, 100 particles): ~30-60 minutes

---

## Contributing

Contributions are welcome! Please feel free to submit a Pull Request.

### Areas for Contribution
- Additional sports/competition datasets
- Alternative likelihood models
- Performance optimizations
- Enhanced visualizations
- Documentation improvements

---

## License

[Add your license here - e.g., MIT, Apache 2.0, GPL]

---

## References

### Theoretical Background
- **Particle Filters:** Doucet, A., & Johansen, A. M. (2009). "A tutorial on particle filtering and smoothing."
- **EM Algorithm:** Dempster, A. P., et al. (1977). "Maximum likelihood from incomplete data via the EM algorithm."
- **Skill Rating Models:** Glickman, M. E. (1999). "Parameter estimation in large dynamic paired comparison experiments."

### Similar Projects
- **TrueSkill:** Microsoft's skill rating system for Xbox Live
- **Elo Rating:** Traditional chess rating system
- **Glicko:** Rating system with time-varying uncertainty

---

## Contact

For questions or feedback, please open an issue on GitHub or contact [your contact information].

---

**Note:** This project is for educational and research purposes. Skill ratings are statistical estimates and should be interpreted with appropriate uncertainty.