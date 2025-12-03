import numpy as np

class StimulusHandler:
    def __init__(self, led_timestamps: np.ndarray, patterns: np.ndarray, dt: float, logger):
        self.led_timestamps = led_timestamps
        self.patterns = patterns
        self.dt = dt
        self.logger = logger

        self.stim_blocks = self.detect_stim_blocks()
        self.valid_timestamps = self.flatten_stim_timestamps()
        self.bin_edges = np.arange(self.valid_timestamps[0], self.valid_timestamps[-1] + dt, dt)

    def detect_stim_blocks(self):
        ts = self.led_timestamps
        gap_indices = np.where(np.diff(ts) > 0.1)[0]
        start_indices = np.insert(gap_indices + 1, 0, 0)
        end_indices = np.append(gap_indices, len(ts) - 1)
        return [(ts[start], ts[end] + self.dt) for start, end in zip(start_indices, end_indices)]

    def flatten_stim_timestamps(self):
        ts = []
        for start, end in self.stim_blocks:
            block_mask = (self.led_timestamps >= start) & (self.led_timestamps < end)
            ts_block = self.led_timestamps[block_mask]
            ts.extend(ts_block)
        return np.array(ts)

    def generate_stimulus(self):
        ts_mask = np.isin(self.led_timestamps, self.valid_timestamps)
        stimulus = self.patterns[ts_mask]

        n_bins = len(self.bin_edges) - 1
        stimulus = np.unpackbits(stimulus.astype(np.uint8), axis=1)
        stimulus = stimulus.reshape(stimulus.shape[0], 16, 16)
        stimulus = stimulus.transpose((1, 2, 0))

        stimulus_left_half = np.flip(stimulus[:, 0:8, :], axis=1)
        stimulus_right_half = np.flip(stimulus[:, 8:16, :], axis=1)
        stimulus_frames = np.hstack((stimulus_left_half, stimulus_right_half))

        stim_uniform = np.zeros((stimulus_frames.shape[0], stimulus_frames.shape[1], n_bins), dtype=stimulus_frames.dtype)
        for i in range(n_bins):
            t_bin = self.bin_edges[i]
            frame_idx = np.searchsorted(self.valid_timestamps, t_bin, side='right') - 1
            frame_idx = max(frame_idx, 0)
            stim_uniform[:, :, i] = stimulus_frames[:, :, frame_idx]

        self.logger.info("Stimulus generated.")
        return stim_uniform