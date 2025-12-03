import os
import numpy as np

class ExportManager:
    """
    Handles exporting of analysis results to disk as structured .npz files.
    For single UnitProcessor runs, RecID is included in the filename.
    """

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

    def add_record(self, rec_id, channel: str, sta_array: np.ndarray):
        """
        Add a record (RecID, Channel, STA array) to the export dataset.

        For single RecID exports, we don't store RecID inside STA file — 
        it's already in the filename.
        """
        if not self.enabled:
            return

        num_lags = sta_array.shape[2]
        for lag_idx in range(num_lags):
            self.records.append({
                "Channel": channel,
                "STA": sta_array[:, :, lag_idx],
                "Lag": lag_idx
            })

    def save_npz(self, filename=None, rec_id=None):
        """
        Save accumulated records into a .npz file.
        If `rec_id` is provided, include it in the filename.
        """
        if not self.enabled:
            self.logger.info("Export disabled — skipping save.")
            return

        if not self.records:
            self.logger.warning("No records to export.")
            return

        channels = np.array([r["Channel"] for r in self.records])
        stas = np.array([r["STA"] for r in self.records])  # (#records, H, W)
        lags = np.array([r["Lag"] for r in self.records])

        if rec_id is not None:
            filename = filename or f"sta_export_recid_{rec_id}.npz"
        else:
            filename = filename or "sta_export.npz"

        outpath = os.path.join(self.export_dir, filename)
        np.savez_compressed(outpath,
                            Channel=channels,
                            STA=stas,
                            Lag=lags)

        self.logger.info(f"Exported structured STA dataset to: {outpath}")
        self.logger.info(f"Total records: {len(self.records)}")