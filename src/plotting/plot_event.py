import matplotlib.pyplot as plt

def plot_event(event, ax=None, special_pfcands_size=1, special_pfcands_color="gray"):
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
        return fig

