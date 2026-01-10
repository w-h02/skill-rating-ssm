# Football Skill Rating with Particle Filters and EM

This project implements a **dynamic skill rating system** for football teams using **state-space models**, **particle filters (SMC)**, and an **EM algorithm** to estimate model parameters. It tracks skill evolution over time, estimates match outcome probabilities, and generates visualizations of team skills and EM convergence.

---

## Table of Contents

- [Overview](#overview)  
- [Features](#features)  
- [Installation](#installation)  
- [Data Requirements](#data-requirements)  
- [Usage](#usage)  
- [Model Details](#model-details)  
- [Output & Visualizations](#output--visualizations)  
- [Performance Considerations](#performance-considerations)  
- [References](#references)

---

## Overview

This project estimates football team skills using a **Bayesian state-space model**:

- Teams’ latent skills evolve over time as a **random walk**.
- Match outcomes are modeled using a **Thurstone-Mosteller likelihood**:
  - Win/loss/draw probabilities based on skill differences.
- **Particle filters (SMC)** are used for approximate inference.
- **EM algorithm** estimates skill evolution variance (`σ²`) and observation noise (`σ²_obs`).

---

## Features

- Load and preprocess football match data (CSV format).  
- Filter teams with insufficient matches.  
- Run EM algorithm with particle filters to estimate model parameters.  
- Compute log-likelihood grids for parameter surfaces.  
- Visualize:
  - Skill trajectories over time  
  - Final skill rankings of teams  
  - EM algorithm convergence and log-likelihood surfaces  

---

## Installation

1. Clone the repository:

```bash
git clone https://github.com/w-h002/skill_rating_ssm.git
cd skill_rating_ssm
