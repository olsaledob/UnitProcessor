import os
import sys
import toml

# Add module path
module_path = os.path.abspath(os.path.join('../modules'))
if module_path not in sys.path:
    sys.path.append(module_path)

from unit_processor import UnitProcessorConfig, plot_sta_grid_lag0, setup_unit_logging

# Load config
config_file = "config.toml"
config = UnitProcessorConfig.load(config_file)

# Setup logger
logger = setup_unit_logging(config.log_dir, log_level="INFO")

# Find all sta_export_recid_*.npz files
sta_files = [f for f in os.listdir(config.export_dir) if f.startswith("sta_export_recid_") and f.endswith(".npz")]

if not sta_files:
    logger.warning("No STA export files found in export directory.")
else:
    logger.info(f"Found {len(sta_files)} STA export files.")

# Loop over files and plot
for sta_file in sta_files:
    sta_path = os.path.join(config.export_dir, sta_file)
    logger.info(f"Plotting STA lag 0 grid for: {sta_file}")
    plot_sta_grid_lag0(sta_path, save_plots=True, plot_dir=config.plot_dir, logger=logger)