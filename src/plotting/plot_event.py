import matplotlib.pyplot as plt
import torch
import os
import numpy as np
import matplotlib.colors as mcolors
from matplotlib import cm
from sklearn.metrics import confusion_matrix
from src.plotting.histograms import score_histogram, confusion_matrix_plot
from src.plotting.plot_coordinates import plot_coordinates

def plot_event_comparison(event, ax=None, special_pfcands_size=1, special_pfcands_color="gray"):
    eta_dq = event.matrix_element_gen_particles.eta
    phi_dq = event.matrix_element_gen_particles.phi
    pt_dq = event.matrix_element_gen_particles.pt
    eta = event.pfcands.eta
    phi = event.pfcands.phi
    pt = event.pfcands.pt
    mapping = event.pfcands.pf_cand_jet_idx.int().tolist()
    print("N jets:", len(event.jets))
    genjet_eta = event.genjets.eta
    genjet_phi = event.genjets.phi
    genjet_pt = event.genjets.pt
    if ax is None:
        fig, ax = plt.subplots(1, 2, figsize=(10, 5))
    # plot eta, phi and the size of circles proportional to p_t. The colors should be either gray (if not in mapping) or some other color that 'represents' the identified jet
    colorlist = ["red", "green", "blue", "purple", "orange", "yellow", "black", "pink", "cyan", "brown", "black", "black", "black", "gray"]
    colors = []
    for i in range(len(eta)):
        colors.append(colorlist[mapping[i]])
    colors = np.array(colors)
    is_special = (event.pfcands.pid.abs() < 4)
    #markers = ["." if not is_special[i] else "v" for i in range(len(eta))]
    #ax[0].scatter(eta, phi, s=pt, c=colors)
    ax[0].scatter(eta[is_special], phi[is_special], s=pt[is_special]*special_pfcands_size, c=special_pfcands_color, marker="v")
    ax[0].scatter(eta[~is_special], phi[~is_special], s=pt[~is_special], c=colors[~is_special])
    ax[0].scatter(eta_dq, phi_dq, s=pt_dq, c="red", marker="^", alpha=1.0)
    ax[0].scatter(genjet_eta, genjet_phi, marker="*", s=genjet_pt, c="blue", alpha=1.0)
    #eta_special = event.special_pfcands.eta
    #phi_special = event.special_pfcands.phi
    #pt_special = event.special_pfcands.pt
    #print("N special PFCands:", len(eta_special))
    #ax[0].scatter(eta_special, phi_special, s=pt_special*special_pfcands_size, c=special_pfcands_color, marker="v")
    # "special" PFCands - electrons, muons, photons satisfying certain criteria
    # Display the jets as a circle with R=0.5
    jet_eta = event.jets.eta
    jet_phi = event.jets.phi
    for i in range(len(jet_eta)):
        circle = plt.Circle((jet_eta[i], jet_phi[i]), 0.5, color="red", fill=False)
        ax[0].add_artist(circle)
    ax[0].set_xlabel(r"$\eta$")
    ax[0].set_ylabel(r"$\phi$")
    ax[0].set_title("PFCands with Jets")
    if event.fatjets is not None:
        colors = []
        for i in range(len(eta)):
            colors.append(colorlist[mapping[i]])
        colors = np.array(colors)
        is_special = (event.pfcands.pid.abs() < 4)
        ax[1].scatter(eta[is_special], phi[is_special], s=pt[is_special] * special_pfcands_size,
                      c=colors[is_special], marker="v")
        ax[1].scatter(eta[~is_special], phi[~is_special], s=pt[~is_special], c=colors[~is_special])
        ax[1].scatter(eta_dq, phi_dq, s=pt_dq, c="red", marker="^", alpha=1.0)
        ax[1].scatter(genjet_eta, genjet_phi, marker="*", s=genjet_pt, c="blue", alpha=1.0)
        ax[1].set_xlabel(r"$\eta$")
        ax[1].set_ylabel(r"$\phi$")
        ax[1].set_title("PFCands with FatJets")
        # Plot the fatjets as a circle with R=0.8 around the center of the fatjet
        fatjet_eta = event.fatjets.eta
        fatjet_phi = event.fatjets.phi
        fatjet_R = 0.8
        for i in range(len(fatjet_eta)):
            circle = plt.Circle((fatjet_eta[i], fatjet_phi[i]), fatjet_R, color="red", fill=False)
            ax[1].add_artist(circle)
            # even aspect ratio
            ax[1].set_aspect("equal")
        ax[0].set_aspect("equal")
    if ax is not None:
        fig.tight_layout()
        return fig


def plot_event(event, colors="gray", custom_coords=None, ax=None, jets=True):
    # plots event onto the specified ax.
    # :colors: color of the pfcands
    # :colors_special: color of the special pfcands
    # :ax: matplotlib ax object to plot onto
    # :custom_coords: Plot eta and phi from custom_coords instead of event.pfcands.
    make_fig = ax is None
    if ax is None:
        fig, ax = plt.subplots(1, 1, figsize=(5, 5))
    eta_dq = event.matrix_element_gen_particles.eta
    phi_dq = event.matrix_element_gen_particles.phi
    pt_dq = event.matrix_element_gen_particles.pt
    eta = event.pfcands.eta
    phi = event.pfcands.phi
    pt = event.pfcands.pt
    #eta_special = event.special_pfcands.eta
    #phi_special = event.special_pfcands.phi
    #pt_special = event.special_pfcands.pt
    if custom_coords:
        eta = custom_coords[0]
        phi = custom_coords[1]
        #if len(eta_special):
        #    eta_special = eta[-len(eta_special):]
        #    phi_special = phi[-len(phi_special):]
        #    eta = eta[:-len(eta_special)]
        #    phi = phi[:-len(eta_special)]
    genjet_eta = event.genjets.eta
    genjet_phi = event.genjets.phi
    genjet_pt = event.genjets.pt
    #if len(eta_special):
    #    colors_special = colors[-len(eta_special):]
    #    colors = colors[:-len(eta_special)]
    #    print("Colors_special", colors_special)
    #    assert len(colors) == len(phi)
    #    assert len(colors_special) == len(eta_special)
    ax.scatter(eta, phi, s=pt, c=colors)
    ax.scatter(eta_dq, phi_dq, s=pt_dq, c="red", marker="^", alpha=1.0) # Dark quarks
    ax.scatter(genjet_eta, genjet_phi, marker="*", s=genjet_pt, c="blue", alpha=1.0)
    #if len(eta_special):
    #    ax.scatter(eta_special, phi_special, s=pt_special, c=colors_special, marker="v")
    if jets:
        jet_eta = event.fatjets.eta
        jet_phi = event.fatjets.phi
        for i in range(len(jet_eta)):
            circle = plt.Circle((jet_eta[i], jet_phi[i]), 0.8, color="red", fill=False)
            ax.add_artist(circle)
    ax.set_xlabel(r"$\eta$")
    ax.set_ylabel(r"$\phi$")
    ax.set_aspect("equal")
    if make_fig:
        fig.tight_layout()
        return fig


def get_idx_for_event(obj, i):
    return obj.batch_number[i], obj.batch_number[i+1]

def get_labels_jets(b, pfcands, jets):
    # b: Batch of events
    R = 0.8
    labels = torch.zeros(len(pfcands)).long()
    for i in range(len(b)):
        s, e = get_idx_for_event(jets, i)
        dq_eta = jets.eta[s:e]
        dq_phi = jets.phi[s:e]
        if s == e:
            continue
        s, e = get_idx_for_event(pfcands, i)
        pfcands_eta = pfcands.eta[s:e]
        pfcands_phi = pfcands.phi[s:e]
        # calculate the distance matrix between each dark quark and pfcands
        dist_matrix = torch.cdist(
            torch.stack([dq_eta, dq_phi], dim=1),
            torch.stack([pfcands_eta, pfcands_phi], dim=1),
            p=2
        )
        dist_matrix = dist_matrix.T
        closest_quark_dist, closest_quark_idx = dist_matrix.min(dim=1)
        closest_quark_idx[closest_quark_dist > R] = -1
        labels[s:e] = closest_quark_idx
    return (labels >= 0).float()


def plot_batch_eval_OC(event_batch, y_true, y_pred, batch_idx, filename, args, batch):
    # Plot the batch, together with nice colors with object condensation GT and betas
    max_events = 5
    sz = 10
    if args.beta_type == "pt+bc":
        n_columns = 6
        y_true_bc = (y_true >= 0).int()
        score_histogram(y_true_bc, y_pred[:, 3]).savefig(os.path.join(os.path.dirname(filename), "binary_classifier_scores.pdf"))
        score_histogram(y_true_bc, (event_batch.pfcands.pf_cand_jet_idx >= 0).float()).savefig(
            os.path.join(os.path.dirname(filename), "binary_classifier_scores_AK8.pdf"))
        score_histogram(y_true_bc, get_labels_jets(event_batch, event_batch.pfcands, event_batch.fatjets)).savefig(
            os.path.join(os.path.dirname(filename), "binary_classifier_scores_radius_FatJets.pdf"))
        score_histogram(y_true_bc, get_labels_jets(event_batch, event_batch.pfcands, event_batch.genjets)).savefig(
            os.path.join(os.path.dirname(filename), "binary_classifier_scores_radius_GenJets.pdf"))
        fig, ax = plt.subplots(1, 3, figsize=(3*sz/2, sz/2))
        confusion_matrix_plot(y_true_bc, y_pred[:, 3] > 0.5, ax[0])
        ax[0].set_title("Classifier (cut at 0.5)")
        confusion_matrix_plot(y_true_bc, get_labels_jets(event_batch, event_batch.pfcands, event_batch.fatjets), ax[2])
        ax[2].set_title("FatJets")
        confusion_matrix_plot(y_true_bc, get_labels_jets(event_batch, event_batch.pfcands, event_batch.genjets), ax[1])
        ax[1].set_title("GenJets")
        fig.tight_layout()
        fig.savefig(os.path.join(os.path.dirname(filename), "conf_matrices.pdf"))
    else:
        n_columns = 4
    fig, ax = plt.subplots(max_events, n_columns, figsize=(n_columns * sz, sz * max_events))
    # columns: Input coords, colored by beta ; Input coords, colored by GT labels; model coords, colored by beta; model coords, colored by GT labels
    for i in range(event_batch.n_events):
        if i >= max_events:
            break
        event = event_batch[i]
        filt = batch_idx == i
        y_true_event = y_true[filt]
        y_pred_event = y_pred[filt]
        if args.beta_type == "default":
            betas = y_pred_event[:, 3]
        elif args.beta_type == "pt":
            betas = event.pfcands.pt
        elif args.beta_type == "pt+bc":
            betas = event.pfcands.pt
            classifier_labels = y_pred_event[:, 3]
        p_xyz = y_pred_event[:, :3]
        if y_pred_event.shape[1] == 5:
            p_xyz = y_pred_event[:, 1:4]
            e = y_pred_event[:, 0]
            #lorentz_invariant = e**2 - p_xyz.norm(dim=1)**2
            #lorentz_invariant_inputs = event.pfcands.E ** 2 - event.pfcands.pxyz.norm(dim=1) ** 2
        plot_coordinates(event.pfcands.pxyz, pt=event.pfcands.pt, tidx=y_true_event,
                         outdir=os.path.dirname(filename),
                         filename="input_coords_batch_" + str(batch) + "_event_" + str(i) + ".html")
        plot_coordinates(p_xyz, pt=event.pfcands.pt, tidx=y_true_event,
                         outdir=os.path.dirname(filename),
                         filename="model_coords_batch_" + str(batch) + "_event_" + str(i) + ".html")
        y_true_event = y_true_event.tolist()
        clist = ['#1f78b4', '#b3df8a', '#33a02c', '#fb9a99', '#e31a1c', '#fdbe6f', '#ff7f00', '#cab2d6', '#6a3d9a', '#ffff99', '#b15928']
        colors = {
            -1: "gray",
            0: clist[0],
            1: clist[1],
            2: clist[2],
            3: clist[3]
        }
        phi = torch.arctan2(p_xyz[:, 1], p_xyz[:, 0]) #torch.asin(p_xyz[:, 1] / p_xyz.norm(dim=1))
        eta = torch.arctanh(p_xyz[:, 2] / p_xyz.norm(dim=1))
        plot_event(event, colors=plt.cm.brg(betas), ax=ax[i, 0])
        cbar = plt.colorbar(mappable=cm.ScalarMappable(cmap=plt.cm.brg), ax=ax[i, 0]) # How to specify the palette?
        ax[i, 0].set_title(r"input coords, $\beta$ colors")
        cbar.set_label(r"$\beta$")
        plot_event(event, colors=[colors[i] for i in y_true_event], ax=ax[i, 1])
        ax[i, 1].set_title("input coords, GT colors")
        plot_event(event, custom_coords=[eta, phi], colors=plt.cm.brg(betas), ax=ax[i, 2], jets=False)
        #assert betas.min() >= 0 and betas.max() <= 1
        ax[i, 2].set_title(r"model coords, $\beta$ colors")
        cbar = plt.colorbar(mappable=cm.ScalarMappable(cmap=plt.cm.brg), ax=ax[i, 2])
        ax[i, 3].set_title("model coords, GT colors")
        cbar.set_label(r"$\beta$")
        plot_event(event, custom_coords=[eta, phi], colors=[colors[i] for i in y_true_event], ax=ax[i, 3], jets=False)
        if args.beta_type == "pt+bc":
            # Create a custom colormap from light gray to dark green
            colors = [(0.9, 0.9, 0.9), (0.0, 0.5, 0.0)]  # RGB for light gray and dark green
            cmap_name = "lightgray_to_darkgreen"
            custom_cmap = mcolors.LinearSegmentedColormap.from_list(cmap_name, colors)
            plot_event(event, custom_coords=[eta, phi], colors=custom_cmap(classifier_labels), ax=ax[i, 5], jets=False)
            ax[i, 5].set_title(r"model coords, BC label colors")
            cbar = plt.colorbar(mappable=cm.ScalarMappable(cmap=custom_cmap), ax=ax[i, 5])
            cbar.set_label("Classifier score")
            plot_event(event, colors=custom_cmap(classifier_labels), ax=ax[i, 4], jets=False)
            ax[i, 4].set_title(r"input coords, BC label colors")
            cbar = plt.colorbar(mappable=cm.ScalarMappable(cmap=custom_cmap), ax=ax[i, 4])
            cbar.set_label("Classifier score")
    print("Saving eval figure to", filename)
    fig.tight_layout()
    fig.savefig(filename)
    fig.clear()
    plt.close(fig)
