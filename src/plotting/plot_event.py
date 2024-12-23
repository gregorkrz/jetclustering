import matplotlib.pyplot as plt
import torch
import os

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
    ax[0].scatter(eta, phi, s=pt, c=colors)
    ax[0].scatter(eta_dq, phi_dq, s=pt_dq, c="red", marker="^", alpha=1.0)
    ax[0].scatter(genjet_eta, genjet_phi, marker="*", s=genjet_pt, c="blue", alpha=1.0)
    eta_special = event.special_pfcands.eta
    phi_special = event.special_pfcands.phi
    pt_special = event.special_pfcands.pt
    print("N special PFCands:", len(eta_special))
    ax[0].scatter(eta_special, phi_special, s=pt_special*special_pfcands_size, c=special_pfcands_color, marker="v")
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
        ax[1].scatter(eta, phi, s=pt, c=colors)
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
    eta_special = event.special_pfcands.eta
    phi_special = event.special_pfcands.phi
    pt_special = event.special_pfcands.pt
    if custom_coords:
        eta = custom_coords[0]
        phi = custom_coords[1]
        eta_special = eta[-len(eta_special):]
        phi_special = phi[-len(phi_special):]
        eta = eta[:-len(eta_special)]
        phi = phi[:-len(eta_special)]
    genjet_eta = event.genjets.eta
    genjet_phi = event.genjets.phi
    genjet_pt = event.genjets.pt
    colors_special = colors[-len(eta_special):]
    colors = colors[:-len(eta_special)]
    ax.scatter(eta, phi, s=pt, c=colors)
    ax.scatter(eta_dq, phi_dq, s=pt_dq, c="red", marker="^", alpha=1.0) # Dark quarks
    ax.scatter(genjet_eta, genjet_phi, marker="*", s=genjet_pt, c="blue", alpha=1.0)
    ax.scatter(eta_special, phi_special, s=pt_special, c=colors_special, marker="v")
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

def plot_batch_eval_OC(event_batch, y_true, y_pred, batch_idx, filename):
    # Plot the batch, together with nice colors with object condensation GT and betas
    fig, ax = plt.subplots(event_batch.n_events, 4, figsize=(20, 5*event_batch.n_events))
    # columns: Input coords, colored by beta ; Input coords, colored by GT labels; model coords, colored by beta; model coords, colored by GT labels
    for i in range(event_batch.n_events):
        event = event_batch[i]
        filt = batch_idx == i
        y_true_event = y_true[filt]
        y_pred_event = y_pred[filt]
        betas = y_pred_event[:, 3]
        p_xyz = y_pred_event[:, :3]
        eta, phi = torch.atan2(p_xyz[:, 1], p_xyz[:, 0]), torch.asin(p_xyz[:, 2] / p_xyz.norm(dim=1))
        plot_event(event, colors=betas, ax=ax[i, 0])
        cbar = plt.colorbar(ax[i, 0].collections[0], ax=ax[i, 0])
        cbar.set_label(r"$\beta$")
        plot_event(event, colors=y_true_event, ax=ax[i, 1])
        plot_event(event, custom_coords=[eta, phi], colors=betas, ax=ax[i, 2])
        cbar = plt.colorbar(ax[i, 2].collections[0], ax=ax[i, 2])
        cbar.set_label(r"$\beta$")
        plot_event(event, custom_coords=[eta, phi], colors=y_true_event, ax=ax[i, 3])
    fig.tight_layout()
    fig.savefig(filename)
