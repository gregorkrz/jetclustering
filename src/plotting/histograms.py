import numpy as np
import matplotlib.pyplot as plt

def score_histogram(scores_true, scores_pred, ax=None, sz=10):
    make_fig = ax is None
    if make_fig:
        fig, ax = plt.subplots(1, 1, figsize=(sz, sz))
    bins = np.linspace(0, 1, 100)
    pos_scores = scores_pred[scores_true == 1]
    neg_scores = scores_pred[scores_true == 0]
    ax.hist(pos_scores, bins=bins, histtype="step", label="Jet", color=(0, 0.5, 0))
    ax.hist(neg_scores, bins=bins, histtype="step", label="Noise", color=(0.6, 0.6, 0.6))
    ax.set_yscale("log")
    ax.set_xlabel("Classifier score")
    ax.legend()
    ax.grid(1)
    if make_fig:
        fig.tight_layout()
        return fig

from sklearn.metrics import confusion_matrix
def confusion_matrix_plot(ytrue, ypred, ax):
    cm = confusion_matrix(ytrue.int(), ypred.int())
    ax.imshow(cm, cmap="Blues")
    ax.set_xlabel("Predicted label")
    ax.set_ylabel("True label")
    ax.set_xticks([0, 1])
    ax.set_yticks([0, 1])
    ax.set_xticklabels(["Noise", "Jet"])
    ax.set_yticklabels(["Noise", "Jet"])
    for i in range(2):
        for j in range(2):
            ax.text(j, i, cm[i, j], ha="center", va="center", color="black")

def per_pt_score_histogram(y_true, y_pred, pt):
    pt_bins = [[0, 1], [1, 10], [10, 1000]]
    sz = 4
    fig, ax = plt.subplots(len(pt_bins), 1, figsize=(sz, sz*len(pt_bins)))
    for i, (pt_min, pt_max) in enumerate(pt_bins):
        mask = (pt > pt_min) & (pt < pt_max)
        score_histogram(y_true[mask], y_pred[mask], ax=ax[i])
        ax[i].set_title(f"pt $\in$ ({pt_min}, {pt_max})")
    fig.tight_layout()
    return fig

def plot_roc_curve(y_true, y_pred):
    from sklearn.metrics import roc_curve
    fpr, tpr, _ = roc_curve(y_true, y_pred)
    fig, ax = plt.subplots(1, 1, figsize=(5, 5))
    ax.plot(fpr, tpr)
    ax.set_xlabel("False positive rate")
    ax.set_ylabel("True positive rate")
    ax.set_title("ROC curve")
    ax.grid(1)
    return fig
