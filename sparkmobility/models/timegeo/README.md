## Package Structure

```
organized/
├── __init__.py              # Package initialization
├── config.py                # Configuration class with all parameters
├── workflow.py              # Main workflow orchestration class
├── data_processor.py        # Data processing and alignment
├── utils/                   # Utility classes
│   ├── __init__.py
│   ├── parquet_utils.py     # Parquet file handling
│   ├── file_utils.py        # File and directory operations
│   └── cpp_utils.py         # C++ module handling
├── example_usage.py         # Usage examples
└── README.md               # This file
```

## Quick Start

### Basic Usage

```python
from organized.config import TimeGeoConfig
from organized.workflow import TimeGeoWorkflow

# Create configuration
config = TimeGeoConfig(
    input_parquet_path="/path/to/your/data.parquet"
)

# Create and run workflow
workflow = TimeGeoWorkflow(config)
success = workflow.run()
```

### Advanced Usage with Custom Parameters

```python
config = TimeGeoConfig(
    input_parquet_path="/path/to/your/data.parquet",
    output_base_dir="./custom_results",
    num_cpus=32,
    num_stays_threshold=50,
    min_num_stay=5,
    max_num_stay=5000,
    num_days=7,
    work_prob_weekday=0.90,
    work_prob_weekend=0.20,
    sample_fraction=0.05
)

workflow = TimeGeoWorkflow(config)
success = workflow.run()
```

## Configuration Parameters

The `TimeGeoConfig` class provides all configurable parameters with sensible defaults:

### Input/Output
- `input_parquet_path` (required): Path to input parquet file
- `output_base_dir` (default: "./results"): Base directory for all outputs

### Processing
- `num_cpus` (default: 16): Number of CPUs for parallel processing
- `num_stays_threshold` (default: 15): Minimum stays for frequent user extraction

### C++ Module Parameters
- `min_num_stay` (default: 2): Minimum number of stays
- `max_num_stay` (default: 3000): Maximum number of stays
- `nw_thres` (default: 1.0): Network threshold
- `slot_interval` (default: 600): Time slot interval
- `rho` (default: 0.6): Rho parameter
- `gamma` (default: -0.21): Gamma parameter

### Simulation Parameters
- `work_prob_weekday` (default: 0.829): Work probability on weekdays
- `work_prob_weekend` (default: 0.354): Work probability on weekends
- `num_days` (default: 1): Number of days to simulate
- `reg_prob` (default: 0.846): Regularity probability
- `gmm_group_index` (default: 0): GMM group index

### Other Parameters
- `sample_fraction` (default: 0.02): Fraction for other locations sampling
- `parquet_engine` (default: None): Parquet engine (auto-detected if None)

## Input Data Format

The input Parquet file must contain exactly these four columns:

1. **caid**: A unique identifier for each user (any hashable type)
2. **stay_start_timestamp**: The timestamp when the stay begins
3. **type**: A label for the kind of stay (e.g., "home", "work", "other")
4. **h3_id_region**: An H3 index at resolution 16

Example:
```python
import pandas as pd

# Example input data
data = {
    'caid': ['user1', 'user1', 'user2'],
    'stay_start_timestamp': ['2023-01-01 08:00:00', '2023-01-01 18:00:00', '2023-01-01 09:00:00'],
    'type': ['home', 'work', 'home'],
    'h3_id_region': [8928308280fffff, 8928308281fffff, 8928308282fffff]
}

df = pd.DataFrame(data)
df.to_parquet('input_data.parquet')
```

## Output Structure

The workflow creates the following directory structure:

```
results/
├── Parameters/              # Generated parameters
│   ├── Commuters/
│   └── NonCommuters/
├── SRFiltered_to_SimInput/  # Processed input data
├── Simulation/              # Simulation files
│   ├── Locations/
│   ├── Parameters/
│   ├── Mapped/
│   └── Compressed/
├── Analysis/                # Analysis results
└── figs/                    # Generated plots
    ├── 1-HourlyTripCount.png
    ├── 2-StayDuration_All.png
    ├── 3-TripDistance.png
    ├── 4-numVisitedLocations.png
    ├── 5-LocationRank-User1.png
    └── 6-dept_validation.png
```

## Key Features

### 1. Robust Parquet Handling
- Automatic engine detection (PyArrow/FastParquet)
- Fallback mechanisms for compatibility
- Memory-efficient processing

### 2. Comprehensive Error Handling
- Input validation
- Parameter validation
- Graceful error recovery
- Detailed error messages

### 3. Modular Design
- Separated concerns (data processing, simulation, analysis)
- Reusable utility classes
- Easy to extend and modify

### 4. Production Ready
- Proper logging and status reporting
- Resource management
- Timeout handling
- Backup and recovery options

## Utility Classes

### ParquetHandler
Handles parquet file operations with robust fallback support:
```python
from organized.utils.parquet_utils import ParquetHandler

handler = ParquetHandler()
df = handler.read_parquet("data.parquet")
handler.write_parquet(df, "output.parquet")
```

### FileHandler
Provides file and directory utilities:
```python
from organized.utils.file_utils import FileHandler

FileHandler.ensure_dir("./output")
size = FileHandler.get_file_size("file.txt")
```

### CppModuleHandler
Manages C++ module operations:
```python
from organized.utils.cpp_utils import CppModuleHandler

cpp_handler = CppModuleHandler()
success = cpp_handler.run_parameter_generation(
    input_path="data.parquet",
    output_dir="./parameters",
    commuter_mode=False
)
```

### DataProcessor
Handles data processing and validation:
```python
from organized.data_processor import DataProcessor

processor = DataProcessor(parquet_handler)
validation = processor.validate_input_data("data.parquet")
summary = processor.get_data_summary("data.parquet")
```

## Examples

See `example_usage.py` for complete usage examples including:

1. **Basic usage** with default parameters
2. **Custom configuration** with specific parameters
3. **High-performance setup** for large datasets
4. **Error handling** examples
