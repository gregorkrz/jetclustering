import matplotlib.pyplot as plt
import numpy as np


def matrix_plot(result, color_scheme, cbar_label, ax=None, metric_comp_func=None):
    make_fig = ax is None
    dark_masses = [20]
    if make_fig:
        fig, ax = plt.subplots(len(dark_masses), 1, figsize=(5, 5))
    mediator_masses = sorted(list(result.keys()))
    r_invs = sorted(list(set([rinv for mMed in result for mDark in result[mMed] for rinv in result[mMed][mDark]])))
    if len(dark_masses) == 1:
        ax = [ax]
    for i, mDark in enumerate(dark_masses):
        data = np.zeros((len(mediator_masses), len(r_invs)))
        for j, mMed in enumerate(mediator_masses):
            for k, rinv in enumerate(r_invs):
                if mMed not in result:
                    continue
                if mDark not in result[mMed]:
                    continue
                if rinv not in result[mMed][mDark]:
                    continue

                r = result[mMed][mDark][rinv]
                if metric_comp_func is not None:
                    r = metric_comp_func(r)
                data[j, k] = r
        ax[i].imshow(data, cmap="Blues")
        for (j, k), val in np.ndenumerate(data):
            ax[i].text(k, j, f'{val:.3f}', ha='center', va='center', color='black')
        ax[i].set_xticks(range(len(r_invs)))
        ax[i].set_xticklabels(r_invs)
        ax[i].set_yticks(range(len(mediator_masses)))
        ax[i].set_yticklabels(mediator_masses)
        ax[i].set_xlabel("$r_{inv}$")
        ax[i].set_ylabel("$m_{Z'}$ [GeV]")
        #ax[i].set_title(f"mDark = {mDark} GeV")
        cbar = plt.colorbar(ax[i].imshow(data, cmap=color_scheme), ax=ax[i])
        cbar.set_label(cbar_label)
    if make_fig:
        fig.tight_layout()
        return fig

def scatter_plot(ax, xs, ys, label, color=None, pattern=".--"):
    idx = np.argsort(xs)
    xs = np.array(xs)[idx]
    ys = np.array(ys)[idx]
    if color is not None:
        ax.plot(xs, ys, pattern, label=label, color=color)
    else:
        ax.plot(xs, ys, pattern, label=label, color=color)
