import os
import re
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
    def __init__(self, data_dict, led_data, rec_id=None, light_prob=None, config_file='config.toml', log_level="INFO"):
        # Load config and logger
        self.config = UnitProcessorConfig.load(config_file)
        self.logger = setup_unit_logging(self.config.log_dir, log_level)
        self.rec_id = rec_id
        self.light_prob = light_prob
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
        self.stimulus_statistics()
        
        # Export manager
        self.export_manager = ExportManager(
            export_dir=self.config.export_dir,
            logger=self.logger,
            enabled=self.config.export_enabled
        )

    def stimulus_statistics(self, rta_draws=10000):
        stim = self.stimulus

        mean_emp = stim.mean()
        std_emp = stim.std(ddof=1)

        # remove fully dark frames
        frame_sum = stim.sum(axis=(0,1))
        mask_nonzero = frame_sum > 0
        stim_nodark = stim[:,:,mask_nonzero]

        n_dark = np.sum(~mask_nonzero)
        n_non_dark = np.sum(mask_nonzero)
        dark_ratio = n_dark / (n_dark + n_non_dark) if (n_dark + n_non_dark) > 0 else np.nan

        mean_nodark = stim_nodark.mean() if stim_nodark.size > 0 else np.nan
        std_nodark = stim_nodark.std(ddof=1) if stim_nodark.size > 0 else np.nan



        p = self.light_prob
        std_theory = None
        rta_mean = None
        rta_std = None
        rta_error = None

        if p is not None:
            std_theory = np.sqrt(p * (1 - p))

        # RTA
        try:
            rng = np.random.default_rng()
            T = stim.shape[2]

            idx = rng.choice(T, size=rta_draws, replace=True)
            rta_frames = stim[:, :, idx]

            rta = rta_frames.mean(axis=2)

            rta_mean = rta.mean()
            rta_std = rta_frames.std(ddof=1)

        except Exception as e:
            rta_error = str(e)

        # consecutive on frames
        consecutive_on = np.mean(stim[:, :, 1:] * stim[:, :, :-1])

        if p is not None:
            expected_consecutive = p ** 2
        else:
            expected_consecutive = mean_emp ** 2

        # pattern sequence blocks
        frames = stim.reshape(-1, stim.shape[2]).T  # (T, pixels)

        unique_frames = np.unique(frames, axis=0).shape[0]
        total_frames = frames.shape[0]
        repeat_ratio = unique_frames / total_frames

        self.logger.info(f"Stimulus mean (empirical): {mean_emp:.6f}")
        self.logger.info(f"Stimulus std (empirical): {std_emp:.6f}")

        self.logger.info(f"Stimulus mean (no dark frames): {mean_nodark:.6f}")
        self.logger.info(f"Stimulus std (no dark frames): {std_nodark:.6f}")
        self.logger.info(f"Dark frames: {n_dark} / {n_dark+n_non_dark} (ratio={dark_ratio:.6f})")

        if p is not None:
            self.logger.info(f"Stimulus std (theoretical Bernoulli): {std_theory:.6f}  [p={p}]")

        if rta_error is None:
            self.logger.info(
                f"RTA stats: mean={rta_mean:.6f}, std={rta_std:.6f}  (draws={rta_draws})"
            )
        else:
            self.logger.info(f"RTA computation failed: {rta_error}")

        self.logger.info(
            f"Consecutive ON probability: {consecutive_on:.6f}  "
            f"(expected if independent: {expected_consecutive:.6f})"
        )

        self.logger.info(
            f"Unique stimulus frames: {unique_frames} / {total_frames} "
            f"(repeat_ratio={repeat_ratio:.6f})"
        )

        if mean_nodark > 0:
            self.light_prob = mean_nodark
            self.logger.info(f"Using mean of non-dark frames for light probability: {self.light_prob:.6f}")
        
        elif mean_emp > 0:
            self.light_prob = mean_emp
            self.logger.info(f"Using mean of all frames for light probability: {self.light_prob:.6f}")
        else:
            self.logger.info(f"No empirical value could be estimated, using light probability taken from filename: {self.light_prob:.6f}")


    def process_all_units(self):
        for key in self.data_dict.keys():
            self.process_unit(key)
        
        # Final save
        if self.config.export_enabled:
            ensure_dir(self.config.export_dir)
            self.export_manager.save_npz(rec_id=self.rec_id, dt=self.config.dt)

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
        analysis = RFAnalysis(self.stimulus, spike_train, np.zeros((16, 16)), 'CL', p=self.light_prob)
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
            firing_rate_outside=firing_rate_outside,
            spike_count=n_spikes_in_window,
            stim_duration=inside_duration
        )