import os, sys, re
import toml
import numpy as np

# Add module path
module_path = os.path.abspath(os.path.join('../modules'))
if module_path not in sys.path:
    sys.path.append(module_path)

from unit_processor import load_h5_to_dict, UnitProcessor, UnitProcessorConfig, setup_unit_logging, ExportManager

# Load config
config_file = "config.toml"
config = UnitProcessorConfig.load(config_file)

start = config.dt  # or config.start from st_calculation if needed
h5_dir = config.h5_dir
led_dir = config.led_dir

# Extract RecID from filename
def extract_rec_id(filename):
    match = re.search(r"RecID-(\d+)", filename)
    if match:
        return int(match.group(1))
    return None

# Match files by RecID
h5_files = {extract_rec_id(f): os.path.join(h5_dir, f)
            for f in os.listdir(h5_dir)
            if f.endswith(".h5") and extract_rec_id(f) is not None}

led_files = {extract_rec_id(f): os.path.join(led_dir, f)
             for f in os.listdir(led_dir)
             if (f.endswith(".npz") or f.endswith(".h5")) and extract_rec_id(f) is not None}

common_rec_ids = sorted(set(h5_files.keys()) & set(led_files.keys()))
print(f"Found {len(common_rec_ids)} matching file pairs")

# Create shared logger and export manager
logger = setup_unit_logging(config.log_dir, log_level="INFO")
export_manager = ExportManager(export_dir=config.export_dir, logger=logger, enabled=config.export_enabled)

# Loop over RecIDs and process
for rec_id in common_rec_ids:
    export_manager.records = []
    h5_filename = h5_files[rec_id]
    led_filename = led_files[rec_id]

    logger.info(f"\nProcessing RecID {rec_id}")
    logger.info(f"H5 file:  {os.path.basename(h5_filename)}")
    logger.info(f"LED file: {os.path.basename(led_filename)}")

    # Load spike data
    data_dict = load_h5_to_dict(h5_filename)

    # Load LED pattern data (npz!!)
    led_data = np.load(led_filename, allow_pickle=True)

    # Run analysis for this RecID — pass shared export manager
    processor = UnitProcessor(
        data_dict=data_dict,
        led_data=led_data,
        rec_id=rec_id,
        config_file=config_file,
        log_level="INFO"
    )
    processor.export_manager = export_manager  # override to use shared export manager
    processor.process_all_units()

# Save all results for all RecIDs into one big NPZ file
export_manager.save_npz(filename="all_recids_sta_export.npz")