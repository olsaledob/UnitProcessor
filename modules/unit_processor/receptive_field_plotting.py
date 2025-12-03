import numpy as np
import matplotlib.pyplot as plt
import os
from scipy.stats import norm


def plot_sta_lags(obj, name, show_filter=True, save=False, max_per_row=10):
    """
    Standalone function to plot all lags of an STA together with the ground truth filter.
    Follows good figure practice: consistent sizing, colorblind-safe colormap, clean formatting.
    
    Parameters
    ----------
    obj : RFAnalysis-like object
        Any object with attributes `n_lags`, `start`, `sta`, `filter`, and `plot_folder`.
    name : str
        Name for saving output file.
    show_filter : bool, default=True
        Whether to include the filter plot after lags.
    save : bool, default=False
        Whether to save the figure as a PDF in obj.plot_folder.
    max_per_row : int, default=10
        Maximum number of plots per row before wrapping.
    """
    plt.rcParams.update({
                    'font.family': 'Arial',
                    'axes.titlesize': 20,
                    'axes.labelsize': 18,
                    'xtick.labelsize': 16,
                    'ytick.labelsize': 16,
                    'legend.fontsize': 18,
                    'axes.linewidth': 0.8,
                    'figure.dpi': 300,
                    'figure.titlesize': 16,
                    'grid.alpha': 0.4,
                    'legend.loc': "upper right",
                })
    # Colorblind-safe diverging colormap
    cmap = 'seismic'  # safer than 'seismic'; could use 'cividis' for sequential

    total_plots = obj.n_lags + int(show_filter)
    n_rows = int(np.ceil(total_plots / max_per_row))
    n_cols = min(total_plots, max_per_row)

    # Figure size tuned for readability
    fig, axs = plt.subplots(n_rows, n_cols, figsize=(2.5 * n_cols, 3 * n_rows))
    axs = np.array(axs).reshape(-1)  # Flatten for simple indexing

    # Plot each lag
    for i_lag in range(obj.n_lags):
        im = axs[i_lag].imshow(obj.sta[:, :, i_lag], vmin=-1, vmax=1, cmap=cmap)
        axs[i_lag].set_title(f"Lag {i_lag + obj.start}")
        axs[i_lag].set_xticks([])
        axs[i_lag].set_yticks([])

    # Plot filter if requested
    if show_filter:
        im = axs[obj.n_lags].imshow(obj.filter, vmin=-1, vmax=1, cmap=cmap)
        axs[obj.n_lags].set_title("Filter")
        axs[obj.n_lags].set_xticks([])
        axs[obj.n_lags].set_yticks([])

    # Hide unused axes
    for ax in axs[total_plots:]:
        ax.axis("off")

    fig.tight_layout(rect=[0,0,1.2,1])
    # Shared colorbar
    cbar = fig.colorbar(im, ax=axs[:total_plots])
    cbar.set_label("Magnitude")

    # Tight layout for whitespace control
    fig.tight_layout()

    # Save option
    if save:
        out_path = os.path.join(obj.plot_folder, f'sta_lags_{name}.pdf')
        fig.savefig(out_path, dpi=300, bbox_inches='tight')
        print(f"Figure saved to {out_path}")

    plt.show()

def plot_sta_lags_z(obj, name, show_filter=True, save=False, max_per_row=10):
    """
    Plot all lags of a Z-scored STA together with the ground truth filter.
    Uses obj.sta_z (computed during STA calculation) where each value is
    standardized by the mean and std-dev of the raw stimulus ensemble.
    
    Parameters
    ----------
    obj : RFAnalysis-like object
        Must have attributes:
        - n_lags : int
        - start  : int
        - sta_z  : ndarray (H, W, n_lags) containing z-scored STA
        - filter : ndarray (H, W)
        - plot_folder : str
    name : str
        Name used for saving output file.
    show_filter : bool, default=True
        Whether to include the filter plot after lags.
    save : bool, default=False
        Whether to save the figure as a PDF (in obj.plot_folder).
    max_per_row : int, default=10
        Max number of plots per row before wrapping.
    """
    # ---- style parameters ----
    plt.rcParams.update({
        'font.family': 'Arial',
        'axes.titlesize': 20,
        'axes.labelsize': 18,
        'xtick.labelsize': 16,
        'ytick.labelsize': 16,
        'legend.fontsize': 18,
        'axes.linewidth': 0.8,
        'figure.dpi': 300,
        'figure.titlesize': 16,
        'grid.alpha': 0.4,
        'legend.loc': "upper right",
    })
    
    # Diverging colormap for z-scores
    cmap = 'seismic'
    vmin, vmax = -3, 3  # typical z-score range; adjust if needed

    # Safety check
    if not hasattr(obj, "sta_z"):
        raise AttributeError("Object has no 'sta_z' attribute. "
                             "Run calc_sta() with z-score computation first.")

    total_plots = obj.n_lags + int(show_filter)
    n_rows = int(np.ceil(total_plots / max_per_row))
    n_cols = min(total_plots, max_per_row)

    fig, axs = plt.subplots(n_rows, n_cols, figsize=(2.5 * n_cols, 3 * n_rows))
    axs = np.array(axs).reshape(-1)

    # Plot each lag using z-scored STA
    for i_lag in range(obj.n_lags):
        z_slice = obj.sta_z[:, :, i_lag]
        max_abs_z = z_slice[np.unravel_index(np.argmax(np.abs(z_slice)), z_slice.shape)]
        im = axs[i_lag].imshow(obj.sta_z[:, :, i_lag], 
                               vmin=vmin, vmax=vmax, cmap=cmap)
        axs[i_lag].set_title(f"Lag {i_lag + obj.start} ({max_abs_z:.2f})")
        axs[i_lag].set_xticks([])
        axs[i_lag].set_yticks([])

    # Plot filter if requested (raw filter, not z-scored!)
    if show_filter:
        im = axs[obj.n_lags].imshow(obj.filter, 
                                    vmin=vmin, vmax=vmax, cmap=cmap)
        axs[obj.n_lags].set_title("Filter")
        axs[obj.n_lags].set_xticks([])
        axs[obj.n_lags].set_yticks([])

    # Hide any unused axes
    for ax in axs[total_plots:]:
        ax.axis("off")

    fig.tight_layout(rect=[0, 0, 1.2, 1])
    
    # Shared colorbar
    cbar = fig.colorbar(im, ax=axs[:total_plots])
    cbar.set_label("Z-score")

    fig.tight_layout()

    # Save if requested
    if save:
        out_path = os.path.join(obj.plot_folder, f'sta_lags_z_{name}.pdf')
        fig.savefig(out_path, dpi=300, bbox_inches='tight')
        print(f"Figure saved to {out_path}")

    plt.show()

def plot_sta_lags_z_sig(obj, name, show_filter=True, save=False, max_per_row=10,
                    alpha=0.1, fdr_correction=False):
    """
    Plot all lags of a Z-scored STA together with ground truth filter,
    marking statistically significant regions based on z-scores.

    Significance mask can be computed with simple thresholding or
    false discovery rate (FDR) correction.

    Parameters
    ----------
    obj : RFAnalysis-like object
        Must have attributes:
        - n_lags : int
        - start  : int
        - sta_z  : ndarray (H, W, n_lags)
        - filter : ndarray (H, W)
        - plot_folder : str
    name : str
        Name used for saving output file.
    show_filter : bool
        Whether to include the filter plot after lags.
    save : bool
        Whether to save the figure as PDF (in plot_folder).
    max_per_row : int
        Max number of plots per row before wrapping.
    alpha : float
        Significance level (two-sided).
    fdr_correction : bool
        If True, applies Benjamini-Hochberg FDR correction over all pixels/lags.
    """
    plt.rcParams.update({
        'font.family': 'Arial',
        'axes.titlesize': 20,
        'axes.labelsize': 18,
        'xtick.labelsize': 16,
        'ytick.labelsize': 16,
        'legend.fontsize': 18,
        'axes.linewidth': 0.8,
        'figure.dpi': 300,
        'figure.titlesize': 16,
        'grid.alpha': 0.4,
        'legend.loc': "upper right",
    })

    cmap = 'seismic'
    vmin, vmax = -3, 3  # symmetric z-score range

    # Safety check
    if not hasattr(obj, "sta_z"):
        raise AttributeError("Object has no 'sta_z' attribute. "
                             "Run STA calculation with z-score first.")

    total_plots = obj.n_lags + int(show_filter)
    n_rows = int(np.ceil(total_plots / max_per_row))
    n_cols = min(total_plots, max_per_row)

    fig, axs = plt.subplots(n_rows, n_cols, figsize=(2.5 * n_cols, 3 * n_rows))
    axs = np.array(axs).reshape(-1)

    # ---- significance ----
    if fdr_correction:
        # Flatten all z-scores across lags for correction
        pvals_all = 2 * (1 - norm.cdf(np.abs(obj.sta_z.flatten())))
        reject_all, _, _, _ = multipletests(pvals_all, alpha=alpha, method='fdr_bh')
        signif_mask_all = reject_all.reshape(obj.sta_z.shape)
    else:
        zthr = norm.ppf(1 - alpha/2)  # two-tailed threshold
        signif_mask_all = np.abs(obj.sta_z) >= zthr

    # Plot each lag
    for i_lag in range(obj.n_lags):
        data = obj.sta_z[:, :, i_lag]
        im = axs[i_lag].imshow(data, vmin=vmin, vmax=vmax, cmap=cmap)
        axs[i_lag].set_title(f"Lag {i_lag + obj.start}")
        axs[i_lag].set_xticks([])
        axs[i_lag].set_yticks([])

        signif_mask = signif_mask_all[:, :, i_lag]
        # Overlay a contour around significant pixels
        if np.any(signif_mask):
            axs[i_lag].contour(signif_mask, levels=[0.5],
                               colors='black', linewidths=0.8)

    # Plot filter if requested (raw filter, not z-scored)
    if show_filter:
        im = axs[obj.n_lags].imshow(obj.filter, vmin=vmin, vmax=vmax, cmap=cmap)
        axs[obj.n_lags].set_title("Filter")
        axs[obj.n_lags].set_xticks([])
        axs[obj.n_lags].set_yticks([])

    # Hide unused axes
    for ax in axs[total_plots:]:
        ax.axis("off")

    fig.tight_layout(rect=[0, 0, 1.2, 1])
    # Shared colorbar
    cbar = fig.colorbar(im, ax=axs[:total_plots])
    cbar.set_label("Z-score")

    fig.tight_layout()

    # Save if requested
    if save:
        postfix = "_fdr" if fdr_correction else "_rawthr"
        out_path = os.path.join(obj.plot_folder, f'sta_lags_z_{name}{postfix}.pdf')
        fig.savefig(out_path, dpi=300, bbox_inches='tight')
        print(f"Figure saved to {out_path}")

    plt.show()

def plot_eigenvals_stc(obj, name, save=False, lag=None, alpha=1):
    """
    Plot the eigenvalues of an STC matrix for a given RFAnalysis-like object.
    Designed for publication-quality output per good figure guidelines.
    
    Parameters
    ----------
    obj : RFAnalysis-like object
        Must have attributes: analysis_type, eigen_results, eigen_results_lag, 
        start, n_lags, plot_folder.
    name : str
        Name to use for the saved file.
    save : bool, default False
        Whether to save the plot as PDF in obj.plot_folder.
    lag : int or None, default None
        Which lag to plot (LS only). If None, plots all lags.
    alpha : float, default 1
        Transparency of plotted points.
    """
    # palette = ["#1A3157", "#e65c46", "#188989"]
    palette = ["#003546", "#077992", "#b22d2e", "#11a2bc", "#ef9d9e", "#e35052"]

    plt.rcParams.update({
        'font.family': 'Arial',
        'axes.titlesize': 20,
        'axes.labelsize': 16,
        'xtick.labelsize': 12,
        'ytick.labelsize': 12,
        'legend.fontsize': 14,
        'axes.linewidth': 0.8,
        'figure.dpi': 300,
        'figure.titlesize': 20,
        'grid.alpha': 0.4,
        'legend.loc': "upper right",
    })

    fig, ax = plt.subplots(figsize=(6, 4))

    if obj.analysis_type == 'CL':
        # Single eigenvalue set
        eigenvals, _ = obj.eigen_results
        idx = np.argsort(eigenvals)[::-1]
        eigenvals = np.array(eigenvals)[idx]
        ax.plot(eigenvals, marker='.', linestyle='None', alpha=alpha, color='tab:blue')
        ax.set_title("STC Eigenvalues (CL)")
    
    elif obj.analysis_type == 'LS':
        if lag is not None:
            eigenvals, _ = obj.eigen_results_lag[lag]
            idx = np.argsort(eigenvals)[::-1]
            eigenvals = np.array(eigenvals)[idx]
            ax.plot(eigenvals, marker='.', linestyle='None', alpha=alpha,
                    label=f'Lag: {obj.start + lag}', color='tab:blue')
            ax.set_title(f"STC Eigenvalues (LS) Lag {obj.start + lag}")
        else:
            # Plot for all lags
            colors = plt.cm.tab10.colors  # colorblind-safe categorical palette
            for lag_idx in range(obj.n_lags):
                eigenvals, _ = obj.eigen_results_lag[lag_idx]
                idx = np.argsort(eigenvals)[::-1]
                vals_sorted = np.array(eigenvals)[idx]
                ax.plot(vals_sorted, marker='.', linestyle='None',
                        label=f'Lag: {obj.start + lag_idx}',
                        alpha=alpha,
                        color=colors[lag_idx % len(colors)])
            ax.set_title("STC Eigenvalues (LS) — All Lags")

    # Label axes
    ax.set_xlabel('Eigenvalue Index')
    ax.set_ylabel('Eigenvalue')
    
    # Grid with subdued alpha
    ax.grid(True, alpha=0.4)
    
    # Legend only if multiple lines
    if len(ax.lines) > 1:
        ax.legend(frameon=False)
    
    fig.tight_layout()

    if save:
        if obj.analysis_type == 'LS' and lag is None:
            filename = f'stc_eigenvals_{name}_alllags.pdf'
        elif obj.analysis_type == 'LS' and lag is not None:
            filename = f'stc_eigenvals_{name}_lag{obj.start+lag}.pdf'
        else:
            filename = f'stc_eigenvals_{name}.pdf'
        out_path = os.path.join(obj.plot_folder, filename)
        fig.savefig(out_path, dpi=300, bbox_inches='tight')
        print(f"Figure saved to {out_path}")

    plt.show()


def plot_eigenvecs_stc(obj, name, vmin=-1, vmax=1, top_n=None, save=False, indices=None):
    """
    Plots the top eigenvectors of the STC matrix for a given RFAnalysis-like object.
    Works for both CL (Single STC) and LS (Lagged STCs) analyses.

    Parameters
    ----------
    obj : RFAnalysis-like object
        Must have attributes depending on analysis_type:
        'CL' requires eigen_results, sta, start, plot_folder
        'LS' requires eigen_results_lag, h, w, n_lags, start, plot_folder
    name : str
        Name used in the saved file(s).
    vmin, vmax : float, default (-1, 1)
        Colormap limits.
    top_n : int or None, default None
        Number of top eigenvectors to plot.
    save : bool, default False
        Whether to save the plots as PDF in obj.plot_folder.
    indices : list of int or None
        Indices of eigenvectors to plot (only in CL mode).
    """

    palette = ["#1A3157", "#e65c46", "#188989"]

    plt.rcParams.update({
        'font.family': 'Arial',
        'axes.titlesize': 16,
        'axes.labelsize': 18,
        'xtick.labelsize': 16,
        'ytick.labelsize': 16,
        'legend.fontsize': 18,
        'axes.linewidth': 0.8,
        'figure.dpi': 300,
        'figure.titlesize': 18,
        'grid.alpha': 0.4
    })

    if obj.analysis_type == 'CL':
        eigenvals, eigenvecs = obj.eigen_results
        h, w, t = obj.sta.shape
        if top_n is None:
            top_n = getattr(obj, "top_n", 1)
        if indices is None:
            indices = range(top_n)

        for n in indices:
            vec3d = eigenvecs[:, n].reshape(h, w, t)

            fig, axes = plt.subplots(1, t, figsize=(3 * t, 3))
            if t == 1:
                axes = [axes]

            for lag in range(t):
                ax = axes[lag]
                im = ax.imshow(vec3d[:, :, lag],
                               cmap='seismic', vmin=vmin, vmax=vmax)
                ax.set_title(f"Lag {lag + obj.start}")
                ax.text(0.05, 0.9,
                        f"EV: {eigenvals[n]:.2f}",
                        transform=ax.transAxes, color='white')
                ax.set_xticks([])
                ax.set_yticks([])

            fig.suptitle(f"Eigenvector {n+1} (Eigenvalue: {eigenvals[n]:.2f})")
            fig.subplots_adjust(top=0.8)

            # Colorbar for the row of plots
            cbar = fig.colorbar(im, ax=axes)
            cbar.set_label("Magnitude")

            if save:
                out_path = os.path.join(obj.plot_folder,
                                        f"eigenvec_{n+1}_{name}.pdf")
                fig.savefig(out_path, dpi=300, bbox_inches='tight')
                print(f"Saved: {out_path}")

            plt.show()

    elif obj.analysis_type == 'LS':
        if top_n is None:
            top_n = 1

        h, w = obj.h, obj.w
        t = obj.n_lags

        for rank in range(top_n):
            fig, axes = plt.subplots(1, t, figsize=(3 * t, 3))
            if t == 1:
                axes = [axes]

            for lag in range(t):
                eigenvals, eigenvecs = obj.eigen_results_lag[lag]
                order = np.argsort(np.abs(eigenvals))[::-1]
                idx = order[rank]
                ev = eigenvals[idx]
                vec2d = eigenvecs[:, idx].reshape(h, w)

                ax = axes[lag]
                im = ax.imshow(vec2d, cmap='seismic', vmin=vmin, vmax=vmax)
                ax.set_title(f"Lag {lag + obj.start} (EV: {ev:.2f})")
                ax.set_xticks([])
                ax.set_yticks([])

            cbar = fig.colorbar(im, ax=axes)
            cbar.set_label("Magnitude")

            if save:
                out_path = os.path.join(obj.plot_folder,
                                        f"eigenvec_{rank+1}_{name}_ls.pdf")
                fig.savefig(out_path, dpi=300, bbox_inches='tight')
                print(f"Saved: {out_path}")

            plt.show()




