import numpy as np
import toml
import numpy.matlib as ml
import matplotlib.pyplot as plt
from numpy.lib.stride_tricks import sliding_window_view
import scipy.linalg
from sklearn.metrics.pairwise import cosine_similarity
import os

class RFAnalysis:
    """
    Receptive-field analysis based on stimulus–response data.

    The class reads analysis parameters from a TOML file and pre-allocates containers for several results (STA, STC, …).
    Public methods (not shown here) compute those quantities.
    
    Parameters:
        stimulus : ndarray, shape (H, W, T)
            Patterns / image stack shown to the neuron.
        spike_train : ndarray, shape (T,)
            Binary or integer spike counts aligned with the third stimulus dimension (time) -> spike_train[t] is the response to stimulus[:, :, t].
        filter : ndarray, shape (H, W)
            Ground truth filter used to generate stimulus and spike train.
        config_file : str 
            Path to the configuration file in TOML format. Default is 'config.toml'.
            Needs at least the following keys:
            - ["sta_parameters"]["start"] and ["sta_parameters"]["end"] 
              (inclusive start index, exclusive end index for temporal lags).

    Attributes:
        h, w, t : int
            Spatial and temporal dimensions of the stimulus.
        start, end : int
            First (inclusive) and last (exclusive) lag used in
            spike-triggered analyses.
        n_lags : int
            Number of lags (end - start)
        sta : None or np.ndarray, shape (H, W, n_lags)
            Spike-triggered average.
        raw_mu : None or np.ndarray, shape (H*W*n_lags,)
            Mean of the raw stimulus matrix.
        stc : None or np.ndarray shape (H*W*n_lags, H*W*n_lags)
            Spike-triggered covariance.
        raw_cov : None or np.ndarray, shape (H*W*n_lags, H*W*n_lags)
            Covariance of the raw stimulus matrix.
        stc_minus_C : None or np.ndarray, shape (H*W*n_lags, H*W*n_lags)
            stc - raw_cov.
        eigen_results : None or tuple of (eigenvalues, eigenvectors)
            Eigen-vectors/values of stc_minus_C.

    Notes:
        After construction call calc_sta_and_stc to fill the result fields.
    """

    def __init__(self, stimulus, spike_train, filter, analysis_type = None, config_file:str='config.toml', p=None):
        """
        Initialize the RFAnalysis class with stimulus, spike_train, filter, analysis type and configuration file. Analysis type can be 'CL' (Cross-lag) or 'LS' (Lag-separate).
        """
        # Read and parse config
        with open(config_file, "r") as f:
            config = toml.load(f)

        self.start = config["st_calculation"]["start"]
        self.end = config["st_calculation"]["end"]
        self.top_n = config["st_plotting"]["top_n"]
        self.n_lags = self.end - self.start
        self.plot_folder = config["st_plotting"]["plot_folder"]

        # Store inputs
        self.stimulus = stimulus
        self.spike_train = spike_train
        self.filter = filter
        self.analysis_type = analysis_type
        self.p = p

        # Derive parameters
        self.h = stimulus.shape[0]  # height
        self.w = stimulus.shape[1]  # width
        self.t = stimulus.shape[2]  # time

        # Matrix buffers
        self.X_raw = None
        self.t_raw = None
        self.X_rt = None
        self.X_st = None

        # Result buffers
        self.sta = None
        self.rta = None
        self.stc = None
        self.raw_mu  = None
        self.raw_cov = None
        self.stc_minus_C = None
        self.eigen_results = None

        # Sanity checks
        self._input_data_check()  # Checks if stimulus, spike_train and filter are compatible
        self._build_matrices()

    def _build_matrices(self):
        """
        Build the matrices for the specified analysis type.
        Currently supports 'PL CL vs LS' and 'RTA'.
        """
        if self.analysis_type == 'CL':
            self.X_raw, self.t_raw = self._build_raw_matrix()
            self.X_st = self._build_spike_triggered_matrix()
        elif self.analysis_type == 'LS':
            self.X_raw, self.t_raw = self._build_raw_cube()
            self.X_st = self._build_spike_triggered_cube(self.X_raw, self.t_raw)


    def calc_rta(self, n_draws = None, seed = None, replace = True, store_idx = False, center = False):
        
        if n_draws is None:
            n_draws = int(self.spike_train.sum())
        
        if n_draws <= 0:
            raise ValueError("Number of draws must be a positive integer.")
        
        if n_draws > self.t and not replace:
            replace = True
            print("Warning: n_draws is larger than the number of time points in the stimulus. Setting replace=True to allow sampling with replacement.")

        rng = np.random.default_rng(seed)
        idx = rng.choice(self.t, size=n_draws, replace=replace)
        rta_frames = self.stimulus[:, :, idx]

        self.rta = np.mean(rta_frames, axis=2)

        if center:  # subtract average stimulus of all frames
            full_mean = np.mean(self.stimulus, axis=2)
            self.rta = self.rta - full_mean
        
        self.rta_std = np.std(rta_frames, axis=2, ddof=1)
        self.rta_var = np.var(rta_frames, axis=2, ddof=1)
        
        full_mean = np.mean(self.stimulus, axis=2)
        full_std  = np.std(self.stimulus,  axis=2, ddof=1)

        with np.errstate(divide='ignore', invalid='ignore'):
            self.z_score = (self.rta - full_mean) / (full_std / np.sqrt(n_draws))
            self.z_score = np.nan_to_num(self.z_score, nan=0.0, posinf=0.0, neginf=0.0)

        if store_idx:
            self.rta_frame_idx = idx

    def _build_raw_matrix(self):
        """
        Build a matrix whose columns are flattened spatio-temporal stimulus slices for each valid reference time t and its lags.

        Attributes used:
            start, end : int 
                Start/End index for the lags.
            h, w : int
                Height, Width of the stimulus.
            n_lags : int
                Number of lags (end - start). 
            t : int
                Total time points in the stimulus/spike train.

        Returns:
            X : np.ndarray, shape (H*W*L, N_raw)
                Every column contains the flattened stimulus slice for a *single* reference time t and its lags (given by start/stop in config).
            t_valid : np.ndarray, shape (N_raw,)
                the reference time indices (0 … T-1) that ended up in the matrix.
        """
        
        # Remapping
        T = self.t 
        stim = self.stimulus  
        lags = np.array(list(range(self.start, self.end)), dtype=int)
        vec_length = self.h * self.w * self.n_lags

        # 1) Build a bool mask
        valid_mask = np.ones(T, dtype=bool)

        # 2) For each lag, set all entries that are before the start or after the end to false
        for lag in lags:
            if lag < 0:
                valid_mask[: -lag] = False
            elif lag > 0:
                valid_mask[T-lag :] = False

        # 3) Collect all valid timestamps t, i.e. those that are not masked out & preallocated the raw matrix
        t_valid = np.nonzero(valid_mask)[0]
        N_raw = t_valid.size  # number of raw samples
        X = np.empty((vec_length, N_raw), dtype=stim.dtype) 

        # 4) Fill the raw matrix
        col = 0
        for t in t_valid:
            frames = stim[:, :, t + lags]  # (H, W, L) shaped array of the stimulus at time t and its lags
            X[:, col] = frames.reshape(-1)  # flatten all frames and store in one column of the raw matrix
            col += 1  # move on to the next column

        return X, t_valid

    def _build_spike_triggered_matrix(self):
        """
        Replicates columns of X_raw according to the number of spikes
        at the corresponding time bin, returning the spike-triggered ensemble.

        Args:
            raw_matrix : np.ndarray, shape (H*W*L, N_raw) 
                Matrix of the raw stimuli
            t_raw : np.ndarray, shape (N_raw,) 
                Valid time indices of raw stimuli

        Attributes used:
            spike_train : np.ndarray (N_spk,)
                1D array of spike counts at each time bin.

        Returns:
            st_matrix (np.ndarray): shape (H*W*L, N_spk)
                Matrix whose columns correspond to *individual spikes*! If the spike_train held 3 spikes in one bin, 
                the respective stimulus vector is repeated three times.
        """
        X_raw = self.X_raw 
        t_raw = self.t_raw
        spike_counts = self.spike_train[t_raw].astype(int)  # Extract spike counts (possibly >1 spike per bin)
        N_spk = spike_counts.sum()  # total num of spikes
        if N_spk == 0:
            raise RuntimeError("No spikes fall into the valid time window; STA/STC can not be computed.")
        D, _ = X_raw.shape
        X_st = np.empty((D, N_spk), dtype=X_raw.dtype)  # allocate

        idx = 0
        for col, n_spk in enumerate(spike_counts):
            if n_spk == 0:
                continue
            X_st[:, idx:idx+n_spk] = np.repeat(X_raw[:, col][:, None], repeats=n_spk, axis=1)  # repeat all stimulus vectors according to the number of spikes
            idx += n_spk
        return X_st

    def _build_raw_cube(self):
        """
        Return raw stimulus as a (D, N_raw, L) cube with L = end-start lags.
        No lag-concatenation, i.e. axis-2 keeps the lags separate.
        """
        T = self.t
        stim2d = self.stimulus.reshape(self.h*self.w, T)  # (D , T)
        lags = np.arange(self.start, self.end)  # shape (L,)
        L = lags.size

        valid = np.ones(T, dtype=bool)
        for lag in lags:
            if lag < 0:
                valid[: -lag] = False
            elif lag > 0:
                valid[T-lag :] = False

        t_valid = np.where(valid)[0]  # (N_raw,)
        N_raw = t_valid.size
        D = self.h * self.w
        
        X = np.empty((D, N_raw, L), dtype=stim2d.dtype)
        for k, lag in enumerate(lags):
            X[:, :, k] = stim2d[:, t_valid + lag]
        return X, t_valid

    def _build_spike_triggered_cube(self, raw_cube, t_valid):
        """
        Replicates raw_cube columns according to spike counts → spike-triggered cube.
        Output shape: (D, N_spk, L).  Keeps lags separate.
        """
        spike_counts = self.spike_train[t_valid].astype(int)
        N_spk = spike_counts.sum()
        if N_spk == 0:
            raise RuntimeError("No spikes in valid window.")
        
        D, _, L = raw_cube.shape
        X_st = np.empty((D, N_spk, L), dtype=raw_cube.dtype)
        idx = 0
        for col, n in enumerate(spike_counts):
            if n == 0:
                continue
            X_st[:, idx:idx+n, :] = np.repeat(raw_cube[:, col, :][:, None, :], n, axis=1)
            idx += n
        return X_st

    def calc_sta(self, center = False):
        """
        Calculate the spike-triggered average (STA) and spike-triggered covariance (STC) from the stimulus and spike train.
        This method builds a raw matrix of the stimulus at the specified lags, computes the spike-triggered ensemble,
        and calculates the STA and STC matrices. It also performs eigen-decomposition on the STC matrix minus the raw covariance using np.linalg.eigh.
        
        Attributes used:
            start, end : int
                Start/End index for the lags.
            h, w : int
                Height, Width of the stimulus.
            n_lags : int
                Number of lags (end - start).
            t : int
                Total time points in the stimulus/spike train.
        Attributes set:
            sta : np.ndarray, shape (H, W, N_lags)
            raw_mu : np.ndarray, shape (H*W*N_lags,)
            stc : np.ndarray, shape (H*W*N_lags, H*W*N_lags)
            raw_cov : np.ndarray, shape (H*W*N_lags, H*W*N_lags)
            stc_minus_C : np.ndarray, shape (H*W*N_lags, H*W*N_lags)
            eigen_results : tuple of (eigenvalues, eigenvectors)
        Raises:
            RuntimeError: If no spikes fall into the valid time window.
        """ 
        if self.analysis_type == 'CL':
            self.sta = self.X_st.mean(axis=1)
            self.sta = self.sta.reshape(self.h, self.w, self.n_lags)
            

            # if we have an actual probabiltiy, use bernoulli:
            if self.p != None:
                if center:    
                    self.sta = self.sta - self.p
            
                                
                # bernoulli normalized
                # the name sta_z is kept, because other code/plotting scripts use it
                std = np.sqrt(self.p * (1-self.p))
                self.sta_z = (self.sta) / std

            else:
                raw_std  = self.X_raw.std(axis=1, ddof=1).reshape(self.h, self.w, self.n_lags)
                raw_mean = self.X_raw.mean(axis=1).reshape(self.h, self.w, self.n_lags)

                if center:    
                    self.sta = self.sta - raw_mean
            
                # z-scored
                self.sta_z = self.sta / raw_std



        elif self.analysis_type == 'LS':
            D, _, L = self.X_raw.shape
            self.sta = np.empty((self.h, self.w, L))

            for k in range(L):
                X_st_k = self.X_st[:, :, k]
                sta_k = X_st_k.mean(axis=1).reshape(self.h, self.w)

                if center:
                    raw_mean = self.X_raw[:, :, k].mean(axis=1).reshape(self.h, self.w)
                    sta_k = sta_k - raw_mean

                self.sta[:, :, k] = sta_k

        # # this approach solves lambda * v = C * v
        # # eigvals, eigvecs = np.linalg.eigh(self.stc_minus_C)
        
        # # most papers seem to use the generalized eigenproblem (A v  = lambda B v)
        # # for this we force symmetry
        # self.raw_cov = 0.5 * (self.raw_cov + self.raw_cov.T)
        # self.stc = 0.5 * (self.stc + self.stc.T)
        # eps = 1e-6 * np.trace(self.raw_cov) / self.raw_cov.shape[0]
        # self.raw_cov += eps * np.eye(self.raw_cov.shape[0])
        # eigvals, eigvecs = scipy.linalg.eigh(self.stc, self.raw_cov, lower=True, check_finite=False)


        # order = np.argsort(np.abs(eigvals))[::-1]
        # self.eigen_results = (eigvals[order], eigvecs[:, order])

    def calc_stc(self, whiten = False, delta = False, eps = 1e-8):

        if self.analysis_type == 'CL':
            X0 = self.X_raw  # (D, N_raw)
            mu = X0.mean(axis=1, keepdims=True)
            X0_centered = X0 - mu

            # raw cov
            C0 = np.cov(X0_centered, bias=False)

            self.raw_mu = mu.squeeze()
            self.raw_cov = C0

            if whiten:
                # ZCA-cor Whitening
                stds = np.sqrt(np.diag(C0))
                V_inv_sqrt = np.diag(1.0 / (stds + eps))

                # Correlation
                P = V_inv_sqrt @ C0 @ V_inv_sqrt

                # eigen-decomp
                eigvals_P, G = np.linalg.eigh(P)
                
                # numeric stability
                eigvals_P[eigvals_P < 0] = 0.0
                P_inv_sqrt = G @ np.diag(1.0 / np.sqrt(eigvals_P + eps)) @ G.T

                # final whitetning
                W_zcacor = P_inv_sqrt @ V_inv_sqrt
                X0 = W_zcacor @ X0_centered
                Xspk = self.X_st - mu
                Xspk = W_zcacor @ Xspk

            else:
                Xspk = self.X_st

            # stc
            Cspk = np.cov(Xspk, bias=False)
            self.stc = Cspk

            # if delta, we subtract the raw cov, if not we simply decompose the stc
            if delta: 
                if whiten:
                    D = X0.shape[0]
                    Cspk -= np.eye(D)
                else:
                    Cspk -= C0

            eigvals, eigvecs = np.linalg.eigh(Cspk)
            
            if delta:
                center = 0
            else:
                center = 1

            order = np.argsort(np.abs(eigvals - center))[::-1]

            self.stc_minus_C = Cspk
            self.eigen_results = (eigvals[order], eigvecs[:, order])


        elif self.analysis_type == 'LS':
            raw_cube = self.X_raw  # (D, N_raw, L)
            spk_cube = self.X_st   # (D, N_spk, L)
            D, N_raw, L = raw_cube.shape

            # allocate
            self.raw_cov_lag = np.empty((L, D, D))
            self.stc_lag = np.empty((L, D, D))
            self.stc_minus_C_lag = np.empty((L, D, D))
            self.eigen_results_lag = []

            for k in range(L):
                X0 = raw_cube[:, :, k]
                mu = X0.mean(axis=1, keepdims=True)
                X0_centered = X0 - mu

                # raw cov for this lag
                C0 = np.cov(X0_centered, bias=False)
                self.raw_cov_lag[k] = C0

                if whiten:
                    # same whitening as CL (ZCA-cor)
                    stds = np.sqrt(np.diag(C0))
                    V_inv_sqrt = np.diag(1.0 / (stds + eps))
                    P = V_inv_sqrt @ C0 @ V_inv_sqrt
                    eigvals_P, G = np.linalg.eigh(P)
                    eigvals_P[eigvals_P < 0] = 0.0
                    P_inv_sqrt = G @ np.diag(1.0 / np.sqrt(eigvals_P + eps)) @ G.T
                    W_zcacor = P_inv_sqrt @ V_inv_sqrt

                    X0 = W_zcacor @ X0_centered
                    Xspk = spk_cube[:, :, k] - mu
                    Xspk = W_zcacor @ Xspk
                else:
                    Xspk = spk_cube[:, :, k]

                Cspk = np.cov(Xspk, bias=False)
                self.stc_lag[k] = Cspk

                if delta:
                    if whiten:
                        Cspk -= np.eye(D)
                    else:
                        Cspk -= C0

                self.stc_minus_C_lag[k] = Cspk

                eigvals, eigvecs = np.linalg.eigh(Cspk)
                if delta:
                    center = 0
                else:
                    center = 1
                order = np.argsort(np.abs(eigvals - center))[::-1]
                self.eigen_results_lag.append((eigvals[order], eigvecs[:, order]))
            
    # Sanity checks

    def _input_data_check(self) -> None:
        """
        Sanity check for the given stimulus, spike_train and filter to ensure they are compatible.

        Args:
            None
        Raises:
            ValueError: If the stimulus is not a 3D array,
                        if the spike_train length does not match the time dimension of the stimulus,
                        or if the end time is before the start time.
        """

        if self.stimulus.ndim != 3:
            raise ValueError("Stimulus must be a 3D array: [height, width, time].")
        if len(self.spike_train) != self.stimulus.shape[2]:
            raise ValueError(f"spike_train length must match the third dimension (time) of stimulus: {len(self.spike_train)} vs {self.stimulus.shape[2]}")
        if self.end < self.start:
            raise ValueError("End must be after start.")
        if self.analysis_type == None:
            raise ValueError("Analysis type must be specified (e.g., 'CL' (Cross-lag) or 'LS' (Lag-separate)).")