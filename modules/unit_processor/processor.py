import os
import numpy as np
from .config import UnitProcessorConfig
from .logging_setup import setup_unit_logging
from .stimulus_handling import StimulusHandler
from .export_manager import ExportManager
from .receptive_field_analysis import RFAnalysis
from .receptive_field_plotting import plot_sta_lags_z

def ensure_dir(path):
    os.makedirs(path, exist_ok=True)

class UnitProcessor:
    def __init__(self, data_dict, led_data, rec_id=None, config_file='config.toml', log_level="INFO"):
        # Load config and logger
        self.config = UnitProcessorConfig.load(config_file)
        self.logger = setup_unit_logging(self.config.log_dir, log_level)
        self.rec_id = rec_id

        # Store spike data
        self.data_dict = data_dict

        # Stimulus preparation
        stim_handler = StimulusHandler(
            led_timestamps=led_data['timestamps'] / 1e6,
            patterns=led_data['patterns'],
            dt=self.config.dt,
            logger=self.logger
        )
        self.stim_blocks = stim_handler.stim_blocks
        self.bin_edges = stim_handler.bin_edges
        self.stimulus = stim_handler.generate_stimulus()

        # Export manager
        self.export_manager = ExportManager(
            export_dir=self.config.export_dir,
            logger=self.logger,
            enabled=self.config.export_enabled
        )

    def process_all_units(self):
        for key in self.data_dict.keys():
            self.process_unit(key)
        
        # Final save
        if self.config.export_enabled:
            ensure_dir(self.config.export_dir)
            self.export_manager.save_npz(rec_id=self.rec_id)

    def process_unit(self, key):
        self.logger.info(f"Processing Unit {key}")
        spike_times = self.data_dict[key]

        stimulus_mask = np.zeros_like(spike_times, dtype=bool)
        for start, end in self.stim_blocks:
            stimulus_mask |= (spike_times >= start) & (spike_times < end)

        spikes_in_stim = spike_times[stimulus_mask]

        if self.config.fraction_spikes < 1.0:
            n_keep = int(np.floor(len(spikes_in_stim) * self.config.fraction_spikes))
            if n_keep < 1:
                self.logger.info(f"{key} skipped -> fraction too small")
                return
            if self.config.seed is not None:
                np.random.seed(self.config.seed)
            spikes_in_stim = np.random.choice(spikes_in_stim, size=n_keep, replace=False)
            spikes_in_stim.sort()

        # Filtering
        n_spikes_in_window = len(spikes_in_stim)
        if n_spikes_in_window <= 1:
            self.logger.info(f"No spikes in stim periods for {key} -> skipped")
            return

        if n_spikes_in_window < self.config.min_spikes_in_window:
            self.logger.info(f"Skipping {key}: only {n_spikes_in_window} spikes during stimulus (min {self.config.min_spikes_in_window})")
            return

        # Firing rates
        inside_duration = sum(end - start for start, end in self.stim_blocks)
        spikes_outside_stim = spike_times[~stimulus_mask]
        outside_duration = (max(spike_times) - min(spike_times)) - inside_duration

        firing_rate_inside = n_spikes_in_window / inside_duration if inside_duration > 0 else 0
        firing_rate_outside = len(spikes_outside_stim) / outside_duration if outside_duration > 0 else 0

        # STA
        spike_train = np.histogram(spikes_in_stim, bins=self.bin_edges)[0]
        analysis = RFAnalysis(self.stimulus, spike_train, np.zeros((16, 16)), 'CL')
        analysis.calc_sta(center=self.config.center)

        if self.config.plotting:
            ensure_dir(self.config.plot_dir)
            plot_sta_lags_z(
                analysis,
                f"Rec-ID_{self.rec_id}_{key}",
                show_filter=False,
                save=self.config.save_plots,
                max_per_row=self.config.max_per_row
            )

        # Add record: RecID, Channel, STA, lags will be handled in ExportManager
        self.export_manager.add_record(
            rec_id=self.rec_id,
            channel=str(key),
            sta_array=analysis.sta_z,
            lag_start=self.config.lag_start,
            firing_rate_inside=firing_rate_inside,
            firing_rate_outside=firing_rate_outside
        )