"""
Browse STAs per UNIT (RecID + Channel): show ALL LAGS together in one figure,
but for paired exports:
  - sta_export_recid_{recid}_dt{dt}s_firstTmins.npz
  - sta_export_recid_{recid}_dt{dt}s_lastTmins.npz

Controls:
- [Enter] or 'n' : next unit
- 'b'            : previous unit
- 'q'            : quit
"""

import os
import sys
import re
import math
import numpy as np
import matplotlib.pyplot as plt

# Settings
MIN_SPIKES = 100

# Optional stim-duration filter; TARGET_MINUTES=None to disable
TARGET_MINUTES = None
TOL_MINUTES = 1.0

MAX_PER_ROW = 10
VMIN, VMAX = -3, 3
CMAP = "seismic"

DT = None  # set to e.g. 0.01 to filter by dt tag in filename, or None for all
DT_TAG = None if DT is None else f"dt{DT:.6g}s"

# Regex for the naming scheme
# sta_export_recid_{recid}_dt{dt}s_(first|last){T}mins.npz
PAT = re.compile(
    r"^sta_export_recid_(?P<recid>\d+)_"
    r"(?P<dt>dt[^_]+)_"
    r"(?P<which>first|last)(?P<tmin>\d+)mins\.npz$"
)

module_path = os.path.abspath(os.path.join("../modules"))
if module_path not in sys.path:
    sys.path.append(module_path)

from unit_processor import UnitProcessorConfig, setup_unit_logging

def load_npz(path):
    d = np.load(path, allow_pickle=True)

    channels = np.array(d["Channel"], dtype=str)
    stas = d["STA"]
    lags = np.array(d["Lag"]).astype(int)

    if "Spike_count" not in d.files:
        raise KeyError("Spike_count")
    spike_count = np.array(d["Spike_count"]).astype(float)

    stim_dur = np.array(d["Stim_duration"]).astype(float) if "Stim_duration" in d.files else None
    recid = np.array(d["RecID"]).astype(int) if "RecID" in d.files else None

    return recid, channels, stas, lags, spike_count, stim_dur


def unit_key(recid, channel):
    return (int(recid), str(channel)) if recid is not None else str(channel)


def _format_unit_title(u):
    meta = []
    if u["recid"] is not None:
        meta.append(f"RecID {u['recid']}")
    meta.append(u["channel"])
    if u["spike_count"] is not None:
        meta.append(f"spikes={u['spike_count']}")
    if u["stim_min"] is not None:
        meta.append(f"stim={u['stim_min']:.1f} min")
    return " | ".join(meta)


def plot_unit_by_source(u, sources_in_order, max_per_row=10, vmin=-3, vmax=3, cmap="seismic"):
    present_sources = [s for s in sources_in_order if s in u["data_by_source"]]
    if not present_sources:
        raise RuntimeError("Unit has no data in any source.")

    block_rows = []
    for s in present_sources:
        n = len(u["data_by_source"][s]["lags"])
        n_cols = min(max_per_row, max(1, n))
        n_rows = int(math.ceil(n / n_cols))
        block_rows.append(n_rows)

    total_rows = sum(block_rows)
    n_cols_global = max_per_row

    fig, axes = plt.subplots(
        total_rows, n_cols_global,
        figsize=(2.2 * n_cols_global, 2.2 * total_rows),
        squeeze=False
    )

    for ax in axes.ravel():
        ax.axis("off")

    im = None
    row0 = 0
    for s, n_rows_block in zip(present_sources, block_rows):
        lags = np.array(u["data_by_source"][s]["lags"], dtype=int)
        stas = np.array(u["data_by_source"][s]["stas"])

        order = np.argsort(lags)
        lags = lags[order]
        stas = stas[order]

        first_ax = axes[row0, 0]
        first_ax.text(
            -0.02, 0.5, s,
            transform=first_ax.transAxes,
            rotation=90, va="center", ha="right", fontsize=9
        )

        for i in range(len(lags)):
            r = row0 + (i // n_cols_global)
            c = (i % n_cols_global)
            ax = axes[r, c]
            ax.axis("on")
            im = ax.imshow(stas[i], vmin=vmin, vmax=vmax, cmap=cmap, origin="lower")
            ax.set_title(f"Lag {lags[i]}", fontsize=9)
            ax.set_xticks([])
            ax.set_yticks([])

        row0 += n_rows_block

    fig.suptitle(_format_unit_title(u), fontsize=12)
    fig.tight_layout(rect=[0.02, 0.02, 1, 0.93])

    if im is not None:
        cbar = fig.colorbar(im, ax=axes.ravel().tolist(), shrink=0.85)
        cbar.set_label("STA (z)")

    return fig


def main():
    config_file = "config.toml"
    config = UnitProcessorConfig.load(config_file)
    logger = setup_unit_logging(config.log_dir, log_level="INFO")

    export_dir = config.export_dir

    # Collect only first/lastTmins exports
    sta_files = []
    for f in os.listdir(export_dir):
        m = PAT.match(f)
        if not m:
            continue
        if DT_TAG is not None and m.group("dt") != DT_TAG:
            continue
        sta_files.append(f)

    sta_files.sort()
    if not sta_files:
        logger.warning("No sta_export_recid_*_dt*_first/last*Tmins.npz files found.")
        return

    target_sec = (TARGET_MINUTES * 60.0) if TARGET_MINUTES is not None else None
    tol_sec = (TOL_MINUTES * 60.0) if TARGET_MINUTES is not None else None

    units = {}
    sources_in_order = []  # labels "first60mins", "last60mins"
    seen_sources = set()

    for fname in sta_files:
        path = os.path.join(export_dir, fname)
        mm = PAT.match(fname)
        which = mm.group("which")
        tmin = mm.group("tmin")
        dt_tag = mm.group("dt")

        source_label = f"{which}{tmin}mins ({dt_tag})"
        if source_label not in seen_sources:
            sources_in_order.append(source_label)
            seen_sources.add(source_label)

        try:
            recid, channels, stas, lags, spk, stim_dur = load_npz(path)
        except KeyError as e:
            logger.warning(f"Skipping {fname}: missing field {e}. Re-export with that field.")
            continue

        for i in range(len(channels)):
            rk = recid[i] if recid is not None else None
            ch = str(channels[i])
            key = unit_key(rk, ch)

            if spk[i] < MIN_SPIKES:
                continue

            if target_sec is not None and stim_dur is not None:
                if abs(stim_dur[i] - target_sec) > tol_sec:
                    continue
            elif target_sec is not None and stim_dur is None:
                continue

            if key not in units:
                units[key] = {
                    "recid": int(rk) if rk is not None else None,
                    "channel": ch,
                    "spike_count": int(round(spk[i])) if np.isfinite(spk[i]) else None,
                    "stim_min": (float(stim_dur[i]) / 60.0) if stim_dur is not None else None,
                    "data_by_source": {},
                }

            if source_label not in units[key]["data_by_source"]:
                units[key]["data_by_source"][source_label] = {"lags": [], "stas": []}

            units[key]["data_by_source"][source_label]["lags"].append(int(lags[i]))
            units[key]["data_by_source"][source_label]["stas"].append(stas[i])

    if not units:
        logger.info(
            f"No units found with Spike_count >= {MIN_SPIKES}"
            + (f" and stim {TARGET_MINUTES}±{TOL_MINUTES} min." if TARGET_MINUTES is not None else ".")
        )
        return

    unit_list = list(units.values())
    unit_list.sort(key=lambda x: (
        x["recid"] if x["recid"] is not None else -1,
        x["channel"]
    ))

    logger.info(f"Found {len(unit_list)} units passing filter (Spike_count >= {MIN_SPIKES}).")
    print("\nControls: [Enter]/n=next, b=back, q=quit\n")

    k = 0
    while 0 <= k < len(unit_list):
        u = unit_list[k]

        fig = plot_unit_by_source(
            u,
            sources_in_order=sources_in_order,
            max_per_row=MAX_PER_ROW,
            vmin=VMIN, vmax=VMAX,
            cmap=CMAP
        )
        fig.text(0.01, 0.01, f"unit {k+1}/{len(unit_list)}", fontsize=9)

        plt.show(block=False)
        cmd = input("Command: ").strip().lower()
        plt.close(fig)

        if cmd == "q":
            break
        elif cmd == "b":
            k = max(0, k - 1)
        else:
            k += 1


if __name__ == "__main__":
    main()