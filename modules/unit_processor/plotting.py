import os
import numpy as np
import matplotlib.pyplot as plt

def plot_channels_rasterplot(data_dict, stim_blocks, led_timestamps, channel_list,
                             rec_id=None, save_plots=False, plot_dir="./plots", logger=None):
    """
    Plot multiple channels' spike events together in a single raster/eventplot.
    This function reproduces the old UnitProcessor.plot_channels_rasterplot behaviour.

    Parameters
    ----------
    data_dict : dict
        Dictionary mapping channel keys to numpy arrays of spike times (in seconds).
    stim_blocks : list of tuples
        Each tuple (start, end) defines a stimulus window in seconds.
    led_timestamps : numpy.ndarray
        LED timestamps (seconds).
    channel_list : list of str
        List of channel keys from data_dict to include in the plot.
    rec_id : int or str, optional
        Recording ID used for the saved filename.
    save_plots : bool
        If True, save the plot as PNG in plot_dir.
    plot_dir : str
        Directory where plots are saved if save_plots is True.
    logger : logging.Logger or None
        Optional logger for warnings/info.
    """

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