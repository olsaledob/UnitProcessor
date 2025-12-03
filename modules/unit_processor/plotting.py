import os
import numpy as np
import matplotlib.pyplot as plt

def plot_channels_rasterplot(data_dict, stim_blocks, led_timestamps, channel_list, rec_id=None, save_plots=False, plot_dir="./plots", logger=None):
    fig_height = max(2, len(channel_list) * 0.6)  # dynamic height
    plt.figure(figsize=(10, fig_height))
    plt.rcParams.update({
        'font.family': 'Arial',
        'axes.titlesize': 16,
        'axes.labelsize': 14,
        'xtick.labelsize': 12,
        'ytick.labelsize': 12,
        'legend.fontsize': 14,
        'axes.linewidth': 0.8,
        'figure.dpi': 300,
        'grid.alpha': 0.4,
        'legend.loc': "upper right",
    })

    # Collect all spike times to determine x-axis max limit
    all_spike_times = []
    for ch in channel_list:
        if ch in data_dict:
            all_spike_times.extend(data_dict[ch])
    if len(all_spike_times) > 0:
        max_time = max(all_spike_times)
    else:
        max_time = max(led_timestamps)

    # Plot stimulus vertical lines
    label_added = False
    for start, end in stim_blocks:
        block_ts = led_timestamps[(led_timestamps >= start) & (led_timestamps < end)]
        if not label_added:
            plt.vlines(block_ts, ymin=-1, ymax=len(channel_list) - 0.5,
                       colors='#e65c46', alpha=0.2, label='Stimulus Window')
            label_added = True
        else:
            plt.vlines(block_ts, ymin=-1, ymax=len(channel_list) - 0.5,
                       colors='#e65c46', alpha=0.2)

    # Label the main stim window position
    min_start = min(s[0] for s in stim_blocks)
    max_end = max(s[1] for s in stim_blocks)
    mid_x = (min_start + max_end) / 2
    mid_y = len(channel_list)
    plt.text(mid_x, mid_y, "Main Stimulus Window",
             ha="center", va="center", fontsize=12,
             bbox=dict(facecolor='white', edgecolor='none', alpha=0.8))

    # Prepare spike trains for plotting
    spike_trains = []
    for ch in channel_list:
        if ch in data_dict:
            spike_trains.append(data_dict[ch])
        else:
            if logger:
                logger.warning(f"Channel {ch} not found in data_dict.")
            spike_trains.append([])

    # Eventplot
    plt.eventplot(spike_trains,
                  lineoffsets=np.arange(len(channel_list)),
                  linelengths=0.4,
                  colors='black',
                  linewidths=0.5,
                  rasterized=True)

    plt.xlabel("Time (s)")
    plt.ylabel("")
    unit_labels = [str(ch).replace("Channel", "Unit") for ch in channel_list]
    plt.yticks(np.arange(len(channel_list)), unit_labels)
    plt.xlim(0, max_time)
    plt.ylim(-0.5, len(channel_list) + 0.5)

    plt.tight_layout()

    # Save if requested
    if save_plots:
        os.makedirs(plot_dir, exist_ok=True)
        filename = f"Rec-ID_{rec_id}_multi_eventplot.png" if rec_id is not None else "multi_eventplot.png"
        outpath = os.path.join(plot_dir, filename)
        plt.savefig(outpath, dpi=600, bbox_inches='tight')
        if logger:
            logger.info(f"Saved multi-channel eventplot to {outpath}")

    plt.show()

def plot_sta_grid_lag0(npz_path, save_plots=False, plot_dir="./plots", logger=None, significance_thresh=0.2):
    cmap = 'seismic'
    vmin, vmax = -3, 3

    data = np.load(npz_path)
    channels = np.array(data["Channel"], dtype=str)
    stas = data["STA"]
    lags = data["Lag"]

    cols = range(1, 9)      # 1..8
    rows = range(8, 0, -1)  # 8..1

    n_rows = len(rows)
    n_cols = len(cols)

    fig, axs = plt.subplots(n_rows, n_cols, figsize=(1.8*n_cols, 1.8*n_rows))
    axs = np.array(axs).reshape(n_rows, n_cols)
    fig.subplots_adjust(wspace=0.05, hspace=0.05)

    # Define corners for annotation
    corner_labels = {
        (0, 0): "Channel_81",
        (0, n_cols-1): "Channel_88",
        (n_rows-1, 0): "Channel_11",
        (n_rows-1, n_cols-1): "Channel_18"
    }

    for r_idx, r in enumerate(rows):
        for c_idx, c in enumerate(cols):
            base_ch = f"Channel_{r}{c}"
            ax = axs[r_idx, c_idx]

            unit_mask = np.array([(ch.startswith(base_ch + "_") and lag == 0)
                                  for ch, lag in zip(channels, lags)])
            unit_indices = np.where(unit_mask)[0]

            if len(unit_indices) == 0:
                _draw_empty_box(ax)
            else:
                # Select most significant unit (max abs z)
                max_val = -np.inf
                chosen_sta = None
                for idx in unit_indices:
                    sta_arr = stas[idx]
                    abs_max = np.max(np.abs(sta_arr))
                    if abs_max > max_val:
                        max_val = abs_max
                        chosen_sta = sta_arr
                if chosen_sta is None or max_val < significance_thresh:
                    _draw_empty_box(ax)
                else:
                    im = ax.imshow(chosen_sta, vmin=vmin, vmax=vmax, cmap=cmap)
                    ax.set_xticks([])
                    ax.set_yticks([])

            # Add corner note if this subplot is in the corner mapping
            if (r_idx, c_idx) in corner_labels:
                ax.text(0.05, 0.05, corner_labels[(r_idx, c_idx)],
                        transform=ax.transAxes,
                        fontsize=8, fontweight="bold",
                        color="black", ha="left", va="bottom",
                        bbox=dict(facecolor="white", edgecolor="none", alpha=0.6, pad=1))

    # Shared colorbar
    cbar = fig.colorbar(im, ax=axs.ravel(), shrink=0.6)
    cbar.set_label("Z-score")

    if save_plots:
        os.makedirs(plot_dir, exist_ok=True)
        outname = os.path.basename(npz_path).replace(".npz", "_lag0_fullgrid.png")
        outpath = os.path.join(plot_dir, outname)
        fig.savefig(outpath, dpi=300, bbox_inches='tight')
        if logger:
            logger.info(f"Saved STA lag0 full grid plot to {outpath}")

    plt.show()


def _draw_empty_box(ax):
    """Draw an empty square box."""
    ax.set_xticks([])
    ax.set_yticks([])
    ax.set_facecolor("white")
    for spine in ax.spines.values():
        spine.set_edgecolor("black")
        spine.set_linewidth(1.0)
    # Blank image so squares are uniform size
    ax.imshow(np.zeros((16, 16)), vmin=0, vmax=0, cmap="Greys")