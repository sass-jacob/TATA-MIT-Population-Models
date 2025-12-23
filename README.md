# TATA-MIT Population Models

Lithium Iron Phosphate particle population-based models

## Description

This repository contains PyBaMM-based models for simulating Lithium Iron Phosphate (LFP) battery particle populations using Fokker-Planck equations and Butler-Volmer kinetics.

## Setup

### Prerequisites

- Python 3.10-3.12 (PyBaMM requires Python < 3.13)
- pip (Python package manager)

### Installation

1. **Clone the repository:**
   ```bash
   git clone <your-repo-url>
   cd TATA-MIT-Population-Models
   ```

2. **Create a virtual environment:**
   ```bash
   python3.12 -m venv venv
   ```

3. **Activate the virtual environment:**
   ```bash
   # On macOS/Linux:
   source venv/bin/activate
   
   # On Windows:
   venv\Scripts\activate
   ```

4. **Install dependencies:**
   ```bash
   pip install -r requirements.txt
   ```

## Usage

After activating the virtual environment, you can run the models:

```bash
# Single population constant current model
python LFP_single_population/single_population_constant_current_butler_volmer.py
```

## Project Structure

- `LFP_single_population/` - Single population models
- `LFP_coupled_populations/` - Coupled population models
- `requirements.txt` - Python dependencies

## Dependencies

Main dependencies include:

- **PyBaMM** - Battery modeling framework
- **NumPy** - Numerical computing
- **Matplotlib** - Plotting and visualization
- **SciPy** - Scientific computing
- **Jupyter** - Interactive notebooks

See `requirements.txt` for a complete list of dependencies and their versions.

## Deactivating the Virtual Environment

When you're done working:

```bash
deactivate
```
