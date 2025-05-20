import matplotlib.pyplot as plt
import numpy as np


def ax_tiny_histogram(ax, labels, colors, values):
    # Create bars
    bars = ax.barh(range(len(labels)), values, color=colors, alpha=0.5)

    # Add labels inside the bars, left-aligned
    for i, (bar, label) in enumerate(zip(bars, labels)):
        ax.text(min(values)-0.004, bar.get_y() + bar.get_height()/2, label,
                va='center', ha='left', fontsize=8, color='black', clip_on=True)
        ax.text(max(values)+0.001, bar.get_y() + bar.get_height()/2,  f'{values[i]:.3f}',
        va='center', ha='right', fontsize=8)

    ax.set_yticks([])
    ax.set_xticks([])  # Hide ticks for minimal look
    for spine in ax.spines.values():
        spine.set_visible(False)

    ax.set_xlim(min(values)-0.005, max(values)+0.005)
    return ax


def multiple_matrix_plot(result, labels, colors, custom_val_formula=lambda x: 2*x[0]*x[1]/(x[0]+x[1]), rename_dict={}): # custom_val_formula set to F1 score, and x is [precision, recall]
    # result: mDark -> mMed -> rinv -> {label->[P, R]} - the order of the labels is set with 'labels' and the colors are set with 'colors'
    # labels: list of labels to plot
    mediator_masses = sorted(list(result.keys()))
    r_invs = sorted(list(set([rinv for mMed in result for rinv in result[mMed]])))
    sz = 3
    #fig, ax = plt.subplots(len(mediator_masses), len(r_invs), figsize=(sz*len(r_invs), 6*len(mediator_masses)))
    fig, ax = plt.subplots(len(mediator_masses), len(r_invs), figsize=(sz*len(r_invs), 0.65*sz*len(mediator_masses)))
    for i, mMed in enumerate(mediator_masses):
        for k, rinv in enumerate(r_invs):
            if mMed not in result:
                continue
            if rinv not in result[mMed]:
                continue
            r = result[mMed][rinv]
            r = {key: custom_val_formula(val) for key, val in r.items()}
            #ax_tiny = fig.add_axes([0.3, 0.1 + i*0.2, 0.15, 0.15])
            #ax_tiny = fig.add_axes([0.1 + k*0.2, 0.1 + i*0.2, 0.15, 0.15])
            for label in labels:
                if label not in r:
                    print("Label not in result:", label , " - skipping!")
                    return None, None
            ax_tiny_histogram(ax[i, k], [rename_dict.get(l,l) for l in labels], colors, [r[label] for label in labels])
            ax[i, k].set_title(f"$m_{{Z'}}$ = {mMed} GeV, $r_{{inv.}}$ = {rinv}")
            #ax.set_title(f"$m_{mMed}$ GeV")
            #ax.set_xlabel("$r_{inv}$")
            #ax.set_ylabel("$m_{Z'}$ [GeV]")
    #ax.set_xticks(range(len(r_invs)))
    #ax.set_xticklabels(r_invs)
    #ax.set_yticks(range(len(mediator_masses)))
    #ax.set_yticklabels(mediator_masses)
    fig.tight_layout()
    return fig, ax

def matrix_plot(result, color_scheme, cbar_label, ax=None, metric_comp_func=None, is_qcd=False):
    make_fig = ax is None
    dark_masses = [20]
    if is_qcd:
        dark_masses = [0]
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
                    try:
                        r = metric_comp_func(r)
                    except:
                        r=0
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
        if color_scheme.lower() == "greens":
            # color it from 0 to 1.0 - set limits on the cbar
            cbar = plt.colorbar(ax[i].imshow(data, cmap=color_scheme), ax=ax[i])
        else:
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
