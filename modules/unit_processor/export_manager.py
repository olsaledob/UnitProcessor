import os
import numpy as np

class ExportManager:
    def __init__(self, export_dir: str, logger, enabled: bool = True):
        self.export_dir = export_dir
        self.logger = logger
        self.enabled = enabled
        self.records = []

        if self.enabled:
            os.makedirs(self.export_dir, exist_ok=True)
            self.logger.info(f"Export directory ready: {self.export_dir}")
        else:
            self.logger.info("Export disabled — no files will be saved.")

    def add_record(self, rec_id, channel: str, sta_array: np.ndarray, lag_start: int, firing_rate_inside=None, firing_rate_outside=None):
        if not self.enabled:
            return
        num_lags = sta_array.shape[2]
        lag_steps = np.arange(lag_start, lag_start + num_lags)
        for pos_idx, lag_step in enumerate(lag_steps):
            self.records.append({
                "Channel": channel,
                "STA": sta_array[:, :, pos_idx],
                "Lag": lag_step,
                "FR_inside": firing_rate_inside,
                "FR_outside": firing_rate_outside
            })

    def save_npz(self, filename=None, rec_id=None):
        if not self.enabled:
            self.logger.info("Export disabled — skipping save.")
            return

        if not self.records:
            self.logger.warning("No records to export.")
            return

        channels = np.array([r["Channel"] for r in self.records])
        stas = np.array([r["STA"] for r in self.records])  # (#records, H, W)
        lags = np.array([r["Lag"] for r in self.records])
        fr_inside = np.array([r["FR_inside"] for r in self.records])
        fr_outside = np.array([r["FR_outside"] for r in self.records])

        if rec_id is not None:
            filename = filename or f"sta_export_recid_{rec_id}.npz"
        else:
            filename = filename or "sta_export.npz"

        outpath = os.path.join(self.export_dir, filename)
        np.savez_compressed(outpath,
            Channel=channels,
            STA=stas,
            Lag=lags,
            FR_inside=fr_inside,
            FR_outside=fr_outside
        )

        self.logger.info(f"Exported structured STA dataset to: {outpath}")
        self.logger.info(f"Total records: {len(self.records)}")