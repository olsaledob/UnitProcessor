# RFArduinoData
This utilizes the framework established in RFAnalysis (https://github.com/fschwar4/sta_analysis) and extends it to experimental data synchronized using the framework in led-logging (https://gitlab.gwdg.de/cadler/led_logging/-/tree/refactor?ref_type=heads) which will soon be replaced by LED_DE_Syncing (https://github.com/olsaledob/LED_DE_Syncing)

## Modules
The modules employed can be found in the 'modules' folder. The main modules are
- load_data (A helper module for loading .h5 / .hdf5 files)
- receptive_field_analysis (used for calculating spike triggered averages (STAs))
- receptive_field_plotting (helping module for plotting)
- unit_processor.py (used to reshape stimulus and experimental data to be compatible with receptive field_analysis)

## Usage
To use the Unit Processor, two datasets are needed:
1. A recordings of spikes (a *_spikesonly.h5 file)
2. Synchronized LED timestamps (a *.npz) file
Note that the filesnames need to contain (unique) RecIDs in order to be paired. Once this has been aquired a UnitProcessor instance can be called:
```python
# Load data
data_dict = load_h5_to_dict(h5_filename)
led_data = np.load(led_filename)

# Generate the instance
processor = UnitProcessor(data_dict, led_data, config_file=config_file, rec_id = rec_id, verbose=True, plotting=False, apply_filters=True, center = False, filename_export = f'export_RecID_{rec_id}')

# Then all units can be processed
processor.process_all_units()
```
The unit processor takes the following optional agruments:
- config_file (str): Containing additional run-time parameters. This is needed but by default the included 'config.toml' is used.
- rec_id (int): Rec_ID used for naming files
- verbose (bool): Determines logging level
- center (bool): If true, an additional centering step is performed. Note: numpy already centers, this then centers once too often
- apply_filters (bool): If true, additional filters can be activated, which currently include exluding all recordings with less than 500 spikes.
- plotting (bool): If true, STA plots are shown. Used to speed up processing if only the exported STAs are important
- filename_export (str): Alternating filename for exported STA files.

Example scripts are located in the 'example_scripts' folder. The jupyter-notebook "analysis.ipynb" demonstrates how to use the unit-processor as described above. Please note that you will have to set the following parameters in the 'config.toml'. Otherwise the code will **not** work on your machine. All folders need to be present, the code does not yet create them by itself.

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

