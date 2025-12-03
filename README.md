# UnitProcessor
The `unit_processor` package provides tools for processing and analysing spike data from experiments, particularly receptive field estimation from LED array stimuli. It utilizes the receptive field analysis framework established in [RFAnalysis](https://github.com/fschwar4/sta_analysis) and extends it to experimental data synchronized using the pipeline in [led-logging](https://gitlab.gwdg.de/cadler/led_logging/-/tree/refactor?ref_type=heads) which is subject to be replaced by [LED_DE_Syncing](https://github.com/olsaledob/LED_DE_Syncing).

## Package Structure
```
unit_processor/
├── __init__.py
├── config.py            # Config dataclass
├── logging_utils.py     # Sets up an overarching logger
├── stimulus.py          # Unpacks and reshapes stimulus
├── processor.py         # Main processing class
├── data_loader.py       # Loading function for HDF5 spike data
├── export_manager.py    # ExportManager for saving STA results
├── plotting.py          # Additional plotting functions
```

## Setup

### Creating the Environment
You can set up the environment with Conda/Miniforge or pip.

**Option 1: Conda/Miniforge**
```bash
conda env create -f environment.yml
conda activate syncenv
```

**Option 2: pip**
```bash
python -m venv venv
source venv/bin/activate  # or venv\Scripts\activate on Windows
pip install -r requirements.txt
```
### Setting up the config
The `config.toml` file controls all paths, processing parameters, logging behaviour, and analysis options

1. **Spike-Triggered Calculation Window**
This section defines the temporal window around each spike for calculating the spike‐triggered average (STA).
```toml
[st_calculation]
start = -9    # time steps before spike for STA
end = 1       # time steps after spike for STA
```

2. **Plotting Settings**
This configures some 
```toml
[st_plotting]
top_n = 3              # used for spike-triggered-covariance (can be ignored)
plot_folder = "plots"  # sub-directory under plot_dir where receptive_field related functions save their results
```

3. **Unit Processor Settings**
This section then defines all parameters necessary for the unit processor.
```toml
[unit_processor]
# Processing parameters
dt = 0.01               # bin size in seconds; defines time-step size for start and end parameters [st_calculation]
save_plots = true       # save generated plots to disk
max_per_row = 5         # max STA plots per row in combined figure
export_enabled = true   # enable exporting STA results

# Paths
h5_dir = "exp_data/h5_halffield"    # directory containing .h5 spike data
led_dir = "exp_data/led_halffield"  # directory containing LED .npz/.h5 files
export_dir = "export"               # where STA export files (.npz) are saved
plot_dir = "plots"                  # where plots are saved
log_dir = "logs"                    # log files for each run

# Analysis options
center = true           # center STA calculation in receptive-field analysis
plotting = true         # enable STA plotting
apply_filters = true    # skip channels with insufficient spikes or poor RF response

# Extra analysis parameters
fraction_spikes = 1.0   # fraction of spikes used in analysis (1.0 = all)
```

### Recommended Project Layout
```
project_root/
├── modules/
│   └── unit_processor/...
├── exp_data/
│   ├── h5_data/           # spike data files (.h5)
│   └── led_data/          # LED pattern files (.npz/.h5)
├── export/                # STA export outputs
├── plots/                 # Plot outputs
├── logs/                  # Log files
├── script.ipynb (or .py)  # own scripts
├── config.toml
├── requirements.txt
└── environment.yml
```

## Example Usage
The usage for a single recording is straight forward. If you want to process multiple files in batch please find the `example.ipynb` in `example_scripts`.
### Single Recording
```python
import numpy as np
from unit_processor import load_h5_to_dict, UnitProcessor

data_dict = load_h5_to_dict("RecID-5_spikesonly.h5")
led_data = np.load("RecID-5_led.npz")

processor = UnitProcessor(data_dict, led_data, rec_id=5, config_file="config.toml")
processor.process_all_units()  # produces sta_export_recid_5.npz
```

```bash
[unit_processor]
h5_dir = "exp_data/h5_halffield"  # Directory for spikesonly files
led_dir = "exp_data/led_halffield" # Directory for synced led logs
export_dir = "export"
plot_dir = "plots"
```

To alter the evaluation behaviour, adjust the following config parameters:
```bash
dt = 0.01  # Timestep in seconds for STA-lags 
save_plots = false  # Save the plots as .pdf
max_per_row = 5  # How many lags are shown per row when plotting
```



## Be aware that...
- because STAs are needed in new parts this structure is likely to change a lot. Especially the unit_processor and receptive_field_analysis modules will likely be reshaped or united to a newer format.
- the config parameters and function arguments are still quite confusing and unorganized. They will be changed in the future.

