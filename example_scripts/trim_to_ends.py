"""
trim_to_ends.py

Processes each matching (H5, LED) RecID pair twice:
  1) First T minutes of the recording (relative to first LED timestamp)
  2) Last T minutes of the recording  (relative to last LED timestamp)

Produces two STA export files:
  - all_recids_sta_export_firstTmin.npz
  - all_recids_sta_export_lastTmin.npz

Please note, this does not shift any timestamps/indices. Events are filtered to the chosen time windows by slicing the LED data.
"""

import os, sys, re
import numpy as np

module_path = os.path.abspath(os.path.join("../modules"))
if module_path not in sys.path:
    sys.path.append(module_path)

from unit_processor import (
    load_h5_to_dict,
    UnitProcessor,
    UnitProcessorConfig,
    setup_unit_logging,
    ExportManager,
)

CONFIG_FILE = "config.toml"

def extract_rec_id(filename: str):
    m = re.search(r"RecID-(\d+)", filename)
    return int(m.group(1)) if m else None

def slice_led_npz_by_time(led_npz: np.lib.npyio.NpzFile, t_start_us: int, t_end_us: int) -> dict:
    """Return a dict with the same keys as the NPZ, but only entries whose timestamps are within [t_start_us, t_end_us]."""
    if "timestamps" not in led_npz:
        raise KeyError("LED npz does not contain key 'timestamps'.")

    ts = np.asarray(led_npz["timestamps"])
    mask = (ts >= t_start_us) & (ts <= t_end_us)

    led_out = {}
    for k in led_npz.files:
        v = led_npz[k]
        try:
            if hasattr(v, "shape") and len(v.shape) >= 1 and v.shape[0] == ts.shape[0]:
                led_out[k] = v[mask]
            else:
                led_out[k] = v
        except Exception:
            led_out[k] = v

    return led_out


def main():
    config = UnitProcessorConfig.load(CONFIG_FILE)
    T_min = int(getattr(config, "trim_window_min", 60))
    dt = float(getattr(config, "dt"))
    ends = str(getattr(config, "trim_ends", "both")).lower()
    if ends not in ("first", "last", "both"):
        raise ValueError(f"config.trim_ends must be one of: first/last/both, got {ends!r}")
    
    WINDOW_US = T_min * 60 * 1_000_000

    h5_dir = config.h5_dir
    led_dir = config.led_dir

    # match files
    h5_files = {
        extract_rec_id(f): os.path.join(h5_dir, f)
        for f in os.listdir(h5_dir)
        if f.endswith(".h5") and extract_rec_id(f) is not None
    }

    led_files = {
        extract_rec_id(f): os.path.join(led_dir, f)
        for f in os.listdir(led_dir)
        if (f.endswith(".npz") or f.endswith(".h5")) and extract_rec_id(f) is not None
    }

    common_rec_ids = sorted(set(h5_files.keys()) & set(led_files.keys()))
    print(f"Found {len(common_rec_ids)} matching file pairs")

    logger = setup_unit_logging(config.log_dir, log_level="INFO")
    export_first = ExportManager(export_dir=config.export_dir, logger=logger, enabled=config.export_enabled)
    export_last  = ExportManager(export_dir=config.export_dir, logger=logger, enabled=config.export_enabled)

    for rec_id in common_rec_ids:
        h5_filename = h5_files[rec_id]
        led_filename = led_files[rec_id]

        logger.info(f"\nProcessing RecID {rec_id}")
        logger.info(f"H5 file:  {os.path.basename(h5_filename)}")
        logger.info(f"LED file: {os.path.basename(led_filename)}")

        # spike, led load
        data_dict = load_h5_to_dict(h5_filename)
        led_npz = np.load(led_filename, allow_pickle=True)
        ts = np.asarray(led_npz["timestamps"])

        if ts.size == 0:
            logger.warning(f"RecID {rec_id}: LED timestamps empty -> skipping.")
            continue
        
        # first and last LED timestamp
        t0 = int(ts[0])
        t1 = int(ts[-1])

        # First window: [t0, t0 + T]
        if ends in ("first", "both"):
            first_start, first_end = t0, t0 + WINDOW_US
            led_first = slice_led_npz_by_time(led_npz, first_start, first_end)
            logger.info(f"RecID {rec_id}: first [{first_start}, {first_end}] -> {len(led_first['timestamps'])} LED events")

            export_first.records = []
            proc_first = UnitProcessor(data_dict=data_dict, led_data=led_first, rec_id=rec_id,
                                       config_file=CONFIG_FILE, log_level="INFO")
            proc_first.export_manager = export_first
            proc_first.process_all_units()

        # Last window: [t1 - T, t1]
        if ends in ("last", "both"):
            last_start, last_end = t1 - WINDOW_US, t1
            led_last = slice_led_npz_by_time(led_npz, last_start, last_end)
            logger.info(f"RecID {rec_id}: last  [{last_start}, {last_end}] -> {len(led_last['timestamps'])} LED events")

            export_last.records = []
            proc_last = UnitProcessor(data_dict=data_dict, led_data=led_last, rec_id=rec_id,
                                      config_file=CONFIG_FILE, log_level="INFO")
            proc_last.export_manager = export_last
            proc_last.process_all_units()

        if ends in ("first", "both"):
            export_first.save_npz(
                filename=f"sta_export_recid_{rec_id}_dt{dt}s_first{T_min}mins.npz"
            )

        if ends in ("last", "both"):
            export_last.save_npz(
                filename=f"sta_export_recid_{rec_id}_dt{dt}s_last{T_min}mins.npz"
            )

if __name__ == "__main__":
    main()
