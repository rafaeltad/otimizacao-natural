# TSP - Traveling Salesman Problem Solver

A Python implementation of the Traveling Salesman Problem (TSP) using Simulated Annealing optimization, specifically designed for Brazilian cities with geographic visualization and MLflow experiment tracking.

## Author
Rafael Tadeu Cardoso dos Santos
UFRJ - COPPE - CPE723

## Features

- **Simulated Annealing Algorithm**: Efficient TSP solver with configurable parameters
- **Geographic Data Integration**: Uses Brazilian geographic data from IBGE and geobr
- **MLflow Integration**: Complete experiment tracking and model management
- **Interactive Visualizations**: Map-based plotting of TSP solutions on Brazilian territory
- **Adaptive Search**: Smart hyperparameter optimization with iterative improvement
- **Configurable Parameters**: JSON-based configuration for easy experimentation

## Installation

### 1. Clone the repository:
```bash
git clone <repository-url>
cd tsp
```

### 2. Set up a virtual environment:

**Using Python venv:**
```bash
python -m venv venv
source venv/bin/activate  # On Windows: venv\Scripts\activate
```

**Using conda:**
```bash
conda create -n tsp-env python=3.13
conda activate tsp-env
```

**Using pyenv:**
```bash
pyenv virtualenv 3.13.3 tsp-env
pyenv activate tsp-env
```

### 3. Install dependencies:
```bash
pip install -r requirements.txt
```

### 4. Development setup (optional):
If you need to modify dependencies, install pip-tools:
```bash
pip install pip-tools
```

## Dependency Management

This project uses `pip-tools` to manage dependencies and ensure reproducible builds:

- **`requirements.in`**: Contains high-level dependencies
- **`requirements.txt`**: Contains pinned versions generated from `requirements.in`

### Updating Dependencies

To add new dependencies:
1. Add them to `requirements.in`
2. Compile the requirements:
```bash
pip-compile requirements.in
```

To update existing dependencies:
```bash
pip-compile --upgrade requirements.in
```

To sync your environment with the exact pinned versions:
```bash
pip-sync requirements.txt
```

## Dependencies

The project uses several key libraries:
- **Optimization**: `scipy`, `scikit-learn`
- **Geographic Data**: `geopandas`, `geobr`, `geodatasets`, `geopy`
- **Visualization**: `matplotlib`, `seaborn`
- **Data Processing**: `pandas`, `numpy`
- **ML Tracking**: `mlflow`
- **Development**: `jupyterlab`, `torch`, `torchvision`

## Usage

### Running the TSP Solver

The main entry point is through the command line:

```bash
python -m tsp config.json
```

### Configuration

Create a JSON configuration file with the following structure:

```json
{
    "hyperparams": {
        "initial_temperature": [1000, 5000, 10000],
        "cooling_rate": [0.95, 0.99],
        "min_temperature": [0.1, 1.0],
        "max_iterations": [1000, 5000]
    },
    "n_cities": 20,
    "mlflow_experiment": "TSP_Experiment",
    "adaptive_search": true,
    "max_search_iterations": 3,
    "cities_data_path": "data/cities.csv",
    "ibge_endpoint": "https://servicodados.ibge.gov.br/api/v3/agregados/6579/periodos/2024/variaveis/9324?localidades=N6[all]"
}
```

## Core Components

### [`tsp.models`](tsp/models.py)
- **`TSPProblem`**: Core TSP problem definition with energy calculation and visualization
- **`SimulatedAnnealing`**: Implementation of the simulated annealing algorithm with MLflow logging

### [`tsp.load`](tsp/load.py)
- Data loading utilities for Brazilian cities and state boundaries
- IBGE API integration for population data
- CSV data processing and GeoDataFrame creation

### [`tsp.utils`](tsp/utils.py)
- Distance calculation using geodesic measurements
- Configuration loading and validation
- Hyperparameter grid search utilities

### [`tsp.__main__`](tsp/__main__.py)
- Main execution logic with adaptive and standard search modes
- MLflow experiment management
- Results logging and best solution tracking

## Key Features

### Geographic Integration
The solver works specifically with Brazilian cities, using:
- IBGE population data
- Geobr state boundaries
- Real geographic coordinates and distances

### MLflow Tracking
Complete experiment tracking including:
- Hyperparameter logging
- Performance metrics
- Solution visualizations as artifacts
- Run comparison and analysis

### Adaptive Search
Intelligent hyperparameter optimization that:
- Learns from previous iterations
- Focuses search on promising parameter regions
- Balances exploration and exploitation

### Visualization
- Map-based TSP solution plotting
- Brazilian state boundaries overlay
- City population visualization
- Route optimization display

## Data Structure

The project expects city data with the following structure:
- `codigo_ibge`: IBGE city code
- `nome`: City name
- `populacao`: Population
- `latitude`: Geographic latitude
- `longitude`: Geographic longitude
- `estado`: State abbreviation

## MLflow Integration

The solver automatically logs:
- All hyperparameters for each run
- Final optimization score
- Solution route and city order
- Geographic plot artifacts
- Search iteration metadata

## Notebooks

The [`tsp/notebooks/`](tsp/notebooks/) directory contains Jupyter notebooks for data exploration, algorithm development, and result analysis. These notebooks provide interactive examples of the TSP solver functionality and data visualization capabilities.

## Docker Support

A [`Dockerfile`](Dockerfile) is provided for containerized deployment:

```bash
docker build -t tsp-solver .
docker run -v $(pwd)/mlruns:/app/mlruns tsp-solver config.json
```

## License

MIT License - see [`LICENSE`](LICENSE) file for details.

## Project Structure

```
tsp/
├── tsp/                    # Main package
│   ├── __main__.py        # Entry point
│   ├── models.py          # TSP and SA implementations
│   ├── load.py            # Data loading utilities
│   ├── utils.py           # Helper functions
│   └── notebooks/         # Jupyter notebooks
├── mlruns/                # MLflow experiment data
├── requirements.txt       # Dependencies
├── requirements.in        # Dependency sources
├── pyproject.toml        # Project configuration
└── README.md             # This file
```

## Development

This project was developed as part of the CPE723 course in the Master's Electrical Engineering Program (PEE) at UFRJ - COPPE, focusing on natural optimization techniques.
