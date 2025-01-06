from typing import Tuple, Union
import numpy as np
import torch
from torch_scatter import scatter_max, scatter_add, scatter_mean
from src.layers.loss_fill_space_torch import LLFillSpace


def safe_index(arr, index):
    # One-hot index (or zero if it's not in the array)
    if index not in arr:
        return 0
    else:
        return arr.index(index) + 1


def assert_no_nans(x):
    """
    Raises AssertionError if there is a nan in the tensor
    """
    if torch.isnan(x).any():
        print(x)
    assert not torch.isnan(x).any()


# FIXME: Use a logger instead of this
DEBUG = False


def debug(*args, **kwargs):
    if DEBUG:
        print(*args, **kwargs)


def calc_LV_Lbeta(
    original_coords,
    g,
    distance_threshold,
    beta: torch.Tensor,
    cluster_space_coords: torch.Tensor,  # Predicted by model
    cluster_index_per_event: torch.Tensor,  # Truth hit->cluster index, e.g. [0, 1, 1, 0, 1, -1, 0, 1, 1]
    batch: torch.Tensor, # E.g. [0, 0, 0, 0, 1, 1, 1, 1, 1]
    # From here on just parameters
    qmin: float = 0.1,
    s_B: float = 1.0,
    noise_cluster_index: int = 0,  # cluster_index entries with this value are noise/noise
    beta_stabilizing="soft_q_scaling",
    huberize_norm_for_V_attractive=False,
    beta_term_option="paper",
    frac_combinations=0,  # fraction of the all possible pairs to be used for the clustering loss
    attr_weight=1.0,
    repul_weight=1.0,
    use_average_cc_pos=0.0,
    loss_type="hgcalimplementation",
    tracking=False,
    dis=False,
    beta_type="default",
    noise_logits=None,
    lorentz_norm=False
) -> Union[Tuple[torch.Tensor, torch.Tensor], dict]:
    """
    Calculates the L_V and L_beta object condensation losses.
    Concepts:
    - A hit belongs to exactly one cluster (cluster_index_per_event is (n_hits,)),
      and to exactly one event (batch is (n_hits,))
    - A cluster index of `noise_cluster_index` means the cluster is a noise cluster.
      There is typically one noise cluster per event. Any hit in a noise cluster
      is a 'noise hit'. A hit in an object is called a 'signal hit' for lack of a
      better term.
    - An 'object' is a cluster that is *not* a noise cluster.
    beta_stabilizing: Choices are ['paper', 'clip', 'soft_q_scaling']:
        paper: beta is sigmoid(model_output), q = beta.arctanh()**2 + qmin
        clip:  beta is clipped to 1-1e-4, q = beta.arctanh()**2 + qmin
        soft_q_scaling: beta is sigmoid(model_output), q = (clip(beta)/1.002).arctanh()**2 + qmin
    huberize_norm_for_V_attractive: Huberizes the norms when used in the attractive potential
    beta_term_option: Choices are ['paper', 'short-range-potential']:
        Choosing 'short-range-potential' introduces a short range potential around high
        beta points, acting like V_attractive.
    Note this function has modifications w.r.t. the implementation in 2002.03605:
    - The norms for V_repulsive are now Gaussian (instead of linear hinge)
    - Noise_logits: If set to an array, it is the output of the noise classifier (whether a particle belongs to a jet or not)
    """
    # remove dummy rows added for dataloader #TODO think of better way to do this
    device = beta.device
    if torch.isnan(beta).any():
        print("There are nans in beta! L198", len(beta[torch.isnan(beta)]))

    beta = torch.nan_to_num(beta, nan=0.0)
    assert_no_nans(beta)
    # ________________________________

    # Calculate a bunch of needed counts and indices locally

    # cluster_index: unique index over events
    # E.g. cluster_index_per_event=[ 0, 0, 1, 2, 0, 0, 1], batch=[0, 0, 0, 0, 1, 1, 1]
    #      -> cluster_index=[ 0, 0, 1, 2, 3, 3, 4 ]
    cluster_index, n_clusters_per_event = batch_cluster_indices(
        cluster_index_per_event, batch
    )
    n_clusters = n_clusters_per_event.sum()
    n_hits, cluster_space_dim = cluster_space_coords.size()
    batch_size = batch.max() + 1
    n_hits_per_event = scatter_count(batch)

    # Index of cluster -> event (n_clusters,)
    batch_cluster = scatter_counts_to_indices(n_clusters_per_event)

    # Per-hit boolean, indicating whether hit is sig or noise
    is_noise = cluster_index_per_event == noise_cluster_index
    is_sig = ~is_noise
    n_hits_sig = is_sig.sum()
    n_sig_hits_per_event = scatter_count(batch[is_sig])
    # Per-cluster boolean, indicating whether cluster is an object or noise
    is_object = scatter_max(is_sig.long(), cluster_index)[0].bool()
    is_noise_cluster = ~is_object

    # FIXME: This assumes noise_cluster_index == 0!!
    # Not sure how to do this in a performant way in case noise_cluster_index != 0
    if noise_cluster_index != 0:
        raise NotImplementedError
    object_index_per_event = cluster_index_per_event[is_sig] - 1
    object_index, n_objects_per_event = batch_cluster_indices(
        object_index_per_event, batch[is_sig]
    )
    n_hits_per_object = scatter_count(object_index)
    # print("n_hits_per_object", n_hits_per_object)
    batch_object = batch_cluster[is_object]
    n_objects = is_object.sum()

    assert object_index.size() == (n_hits_sig,)
    assert is_object.size() == (n_clusters,)
    assert torch.all(n_hits_per_object > 0)
    assert object_index.max() + 1 == n_objects

    # ________________________________
    # L_V term
    # Calculate q
    if beta_type == "default":
        if loss_type == "hgcalimplementation" or loss_type == "vrepweighted":
            q = (beta.clip(0.0, 1 - 1e-4).arctanh() / 1.01) ** 2 + qmin
        elif beta_stabilizing == "paper":
            q = beta.arctanh() ** 2 + qmin
        elif beta_stabilizing == "clip":
            beta = beta.clip(0.0, 1 - 1e-4)
            q = beta.arctanh() ** 2 + qmin
        elif beta_stabilizing == "soft_q_scaling":
            q = (beta.clip(0.0, 1 - 1e-4) / 1.002).arctanh() ** 2 + qmin
        else:
            raise ValueError(f"beta_stablizing mode {beta_stabilizing} is not known")
    elif beta_type  == "pt":
        q = beta
    elif beta_type == "pt+bc":
        q = beta
    assert_no_nans(q)
    assert q.device == device
    assert q.size() == (n_hits,)

    # Calculate q_alpha, the max q per object, and the indices of said maxima
    # assert hit_energies.shape == q.shape
    # q_alpha, index_alpha = scatter_max(hit_energies[is_sig], object_index)
    q_alpha, index_alpha = scatter_max(q[is_sig], object_index)
    assert q_alpha.size() == (n_objects,)

    # Get the cluster space coordinates and betas for these maxima hits too
    x_alpha = cluster_space_coords[is_sig][index_alpha]
    #x_alpha_original = original_coords[is_sig][index_alpha]

    if use_average_cc_pos > 0:
        #! this is a func of beta and q so maybe we could also do it with only q
        x_alpha_sum = scatter_add(
            q[is_sig].view(-1, 1).repeat(1, 3) * cluster_space_coords[is_sig],
            object_index,
            dim=0,
        )  # * beta[is_sig].view(-1, 1).repeat(1, 3)
        qbeta_alpha_sum = scatter_add(q[is_sig], object_index) + 1e-9  # * beta[is_sig]
        div_fac = 1 / qbeta_alpha_sum
        div_fac = torch.nan_to_num(div_fac, nan=0)
        x_alpha_mean = torch.mul(x_alpha_sum, div_fac.view(-1, 1).repeat(1, 3))
        x_alpha = use_average_cc_pos * x_alpha_mean + (1 - use_average_cc_pos) * x_alpha
    if dis:
        phi_sum = scatter_add(
            beta[is_sig].view(-1) * distance_threshold[is_sig].view(-1),
            object_index,
            dim=0,
        )
        phi_alpha_sum = scatter_add(beta[is_sig].view(-1), object_index) + 1e-9
        phi_alpha = phi_sum / phi_alpha_sum

    beta_alpha = beta[is_sig][index_alpha]
    assert x_alpha.size() == (n_objects, cluster_space_dim)
    assert beta_alpha.size() == (n_objects,)


    # Connectivity matrix from hit (row) -> cluster (column)
    # Index to matrix, e.g.:
    # [1, 3, 1, 0] --> [
    #     [0, 1, 0, 0],
    #     [0, 0, 0, 1],
    #     [0, 1, 0, 0],
    #     [1, 0, 0, 0]
    #     ]
    M = torch.nn.functional.one_hot(cluster_index).long()

    # Anti-connectivity matrix; be sure not to connect hits to clusters in different events!
    M_inv = get_inter_event_norms_mask(batch, n_clusters_per_event) - M

    # Throw away noise cluster columns; we never need them
    M = M[:, is_object]
    M_inv = M_inv[:, is_object]
    assert M.size() == (n_hits, n_objects)
    assert M_inv.size() == (n_hits, n_objects)

    # Calculate all norms
    # Warning: Should not be used without a mask!
    # Contains norms between hits and objects from different events
    # (n_hits, 1, cluster_space_dim) - (1, n_objects, cluster_space_dim)
    #   gives (n_hits, n_objects, cluster_space_dim)
    norms = (cluster_space_coords.unsqueeze(1) - x_alpha.unsqueeze(0)).norm(dim=-1)
    assert norms.size() == (n_hits, n_objects)
    L_clusters = torch.tensor(0.0).to(device)
    if frac_combinations != 0:
        L_clusters = L_clusters_calc(
            batch, cluster_space_coords, cluster_index, frac_combinations, q
        )

    # -------
    # Attractive potential term
    # First get all the relevant norms: We only want norms of signal hits
    # w.r.t. the object they belong to, i.e. no noise hits and no noise clusters.
    # First select all norms of all signal hits w.r.t. all objects, mask out later

    if loss_type == "hgcalimplementation" or loss_type == "vrepweighted":
        # if dis:
        #     N_k = torch.sum(M, dim=0)  # number of hits per object
        #     norms = torch.sum(
        #         torch.square(cluster_space_coords.unsqueeze(1) - x_alpha.unsqueeze(0)),
        #         dim=-1,
        #     )
        #     norms_att = norms[is_sig]
        #     norms_att = norms_att / (2 * phi_alpha.unsqueeze(0) ** 2 + 1e-6)
        #     #! att func as in line 159 of object condensation
        #     norms_att = torch.log(
        #         torch.exp(torch.Tensor([1]).to(norms_att.device)) * norms_att + 1
        #     )
     
        N_k = torch.sum(M, dim=0)  # number of hits per object
        if lorentz_norm:
            diff = cluster_space_coords.unsqueeze(1) - x_alpha.unsqueeze(0)
            norms = diff[:, :, 0]**2 - torch.sum(diff[:, :, 1:] ** 2, dim=-1)
            norms = norms.abs() ## ??? Why is this needed? wrong convention?
            #print("Norms", norms[:15])
        else:
            norms = torch.sum(
                torch.square(cluster_space_coords.unsqueeze(1) - x_alpha.unsqueeze(0)),
                dim=-1,
            ) # Take the norm squared
        norms_att = norms[is_sig]
        #! att func as in line 159 of object condensation
        
        norms_att = torch.log(
            torch.exp(torch.Tensor([1]).to(norms_att.device)) * norms_att / 2 + 1
        )

    elif huberize_norm_for_V_attractive:
        norms_att = norms[is_sig]
        # Huberized version (linear but times 4)
        # Be sure to not move 'off-diagonal' away from zero
        # (i.e. norms of hits w.r.t. clusters they do _not_ belong to)
        norms_att = huber(norms_att + 1e-5, 4.0)
    else:
        norms_att = norms[is_sig]
        # Paper version is simply norms squared (no need for mask)
        norms_att = norms_att**2
    assert norms_att.size() == (n_hits_sig, n_objects)

    # Now apply the mask to keep only norms of signal hits w.r.t. to the object
    # they belong to
    norms_att *= M[is_sig]

    # Sum over hits, then sum per event, then divide by n_hits_per_event, then sum over events
    if loss_type == "hgcalimplementation":
        # Final potential term
        # (n_sig_hits, 1) * (1, n_objects) * (n_sig_hits, n_objects)
        # hit_type = (g.ndata["hit_type"][is_sig].view(-1)==3)*4+1  #weight 5 for hadronic hits, 1 for
        # tracks = g.ndata["hit_type"][is_sig]==1
        # hit_type[tracks] = 250
        # total_sum_hits_types = scatter_add(hit_type.view(-1), object_index)
        V_attractive = q[is_sig].unsqueeze(-1) * q_alpha.unsqueeze(0) * norms_att
        assert V_attractive.size() == (n_hits_sig, n_objects)
        #! each shower is account for separately
        V_attractive = V_attractive.sum(dim=0)  # K objects
        #! divide by the number of accounted points
      
        V_attractive = V_attractive.view(-1) / (N_k.view(-1) + 1e-3)
        # V_attractive = V_attractive.view(-1) / (total_sum_hits_types.view(-1) + 1e-3)
        # L_V_attractive = torch.mean(V_attractive)

        ## multiply by a weight that depends on the energy of the shower:
        # print("e_hits", e_hits)
        # print("weight_att", weight_att)
        # L_V_attractive = torch.sum(V_attractive*weight_att)
        L_V_attractive = torch.mean(V_attractive)
        # L_V_attractive = L_V_attractive / torch.sum(weight_att)

        L_V_attractive_2 = torch.sum(V_attractive)
    elif loss_type == "vrepweighted":
        if tracking:
            # weight the vtx hits inside the shower
            V_attractive = (
                g.ndata["weights"][is_sig].unsqueeze(-1)
                * q[is_sig].unsqueeze(-1)
                * q_alpha.unsqueeze(0)
                * norms_att
            )
            assert V_attractive.size() == (n_hits_sig, n_objects)
            V_attractive = V_attractive.sum(dim=0)  # K objects

            L_V_attractive = torch.mean(V_attractive.view(-1))
        else:
            # # weight per hit per shower to compensate for ecal hcal unbalance in hadronic showers
            # ecal_hits = scatter_add(
            #     1 * (g.ndata["hit_type"][is_sig] == 2), object_index
            # )
            # hcal_hits = scatter_add(
            #     1 * (g.ndata["hit_type"][is_sig] == 3), object_index
            # )
            # weights = torch.ones_like(g.ndata["hit_type"][is_sig])
            # weight_ecal_per_object = 1.0 * ecal_hits.clone() + 1
            # weight_hcal_per_object = 1.0 * ecal_hits.clone() + 1
            # mask = (ecal_hits > 2) * (hcal_hits > 2)
            # weight_ecal_per_object[mask] = (ecal_hits + hcal_hits)[mask] / (
            #     2 * ecal_hits
            # )[mask]
            # weight_hcal_per_object[mask] = (ecal_hits + hcal_hits)[mask] / (
            #     2 * hcal_hits
            # )[mask]
            # weights[g.ndata["hit_type"][is_sig] == 2] = weight_ecal_per_object[
            #     object_index
            # ]
            # weights[g.ndata["hit_type"][is_sig] == 3] = weight_hcal_per_object[
            #     object_index
            # ]

            # # weight with an energy log of the hits
            # e_hits = g.ndata["e_hits"][is_sig].view(-1)
            # p_hits = g.ndata["h"][:, -1][is_sig].view(-1)
            # log_scale_s = torch.log(e_hits + p_hits) + 10
            # e_sum_hits = scatter_add(log_scale_s, object_index)
            # # need to take out the weight of alpha otherwise it won't add up to 1
            # e_sum_hits = e_sum_hits - (log_scale_s[index_alpha])
            # e_rel = (log_scale_s) / e_sum_hits[object_index]

            # weight of the hit depending on the radial distance:
            # this weight should help to seed
            # weight_radial_distance = torch.exp(
            #     -g.ndata["radial_distance"][is_sig] / 100
            # )
            # weight_per_object = scatter_add(weight_radial_distance, object_index)
            # weight_radial_distance = (
            #     weight_radial_distance / weight_per_object[object_index]
            # )

            V_attractive = (
                q[is_sig].unsqueeze(-1)  ## weight_radial_distance.unsqueeze(-1)
                * q_alpha.unsqueeze(0)
                * norms_att
            )

            # weight modified showers with a higher weight
            modified_showers = scatter_max(g.ndata["hit_link_modified"], object_index)[
                0
            ]
            n_modified = torch.sum(modified_showers)
            weight_modified = len(modified_showers) / (2 * n_modified)
            weight_unmodified = len(modified_showers) / (
                2 * (len(modified_showers) - n_modified)
            )
            modified_showers[modified_showers > 0] = weight_modified
            modified_showers[modified_showers == 0] = weight_unmodified
            assert V_attractive.size() == (n_hits_sig, n_objects)
            V_attractive = V_attractive.sum(dim=0)  # K objects
            L_V_attractive = torch.sum(
                modified_showers.view(-1) * V_attractive.view(-1)
            ) / len(modified_showers)
    else:
        # Final potential term
        # (n_sig_hits, 1) * (1, n_objects) * (n_sig_hits, n_objects)
        V_attractive = q[is_sig].unsqueeze(-1) * q_alpha.unsqueeze(0) * norms_att
        assert V_attractive.size() == (n_hits_sig, n_objects)
        #! in comparison this works per hit
        V_attractive = (
            scatter_add(V_attractive.sum(dim=0), batch_object) / n_hits_per_event
        )
        assert V_attractive.size() == (batch_size,)
        L_V_attractive = V_attractive.sum()

    # -------
    # Repulsive potential term

    # Get all the relevant norms: We want norms of any hit w.r.t. to
    # objects they do *not* belong to, i.e. no noise clusters.
    # We do however want to keep norms of noise hits w.r.t. objects
    # Power-scale the norms: Gaussian scaling term instead of a cone
    # Mask out the norms of hits w.r.t. the cluster they belong to
    if loss_type == "hgcalimplementation" or loss_type == "vrepweighted":
        if dis:
            norms = norms / (2 * phi_alpha.unsqueeze(0) ** 2 + 1e-6)
            norms_rep = torch.exp(-(norms)) * M_inv
            norms_rep2 = torch.exp(-(norms) * 10) * M_inv
        else:
            norms_rep = torch.exp(-(norms) / 2) * M_inv
            # norms_rep2 = torch.exp(-(norms) * 10) * M_inv
            norms_rep2 = torch.exp(-(norms) * 10) * M_inv
    else:
        norms_rep = torch.exp(-4.0 * norms**2) * M_inv

    # (n_sig_hits, 1) * (1, n_objects) * (n_sig_hits, n_objects)
    V_repulsive = q.unsqueeze(1) * q_alpha.unsqueeze(0) * norms_rep

    # No need to apply a V = max(0, V); by construction V>=0
    assert V_repulsive.size() == (n_hits, n_objects)

    # Sum over hits, then sum per event, then divide by n_hits_per_event, then sum up events
    nope = n_objects_per_event - 1
    nope[nope == 0] = 1
    if loss_type == "hgcalimplementation" or loss_type == "vrepweighted":
        #! sum each object repulsive terms
        L_V_repulsive = V_repulsive.sum(dim=0)  # size number of objects
        number_of_repulsive_terms_per_object = torch.sum(M_inv, dim=0)
        L_V_repulsive = L_V_repulsive.view(
            -1
        ) / number_of_repulsive_terms_per_object.view(-1)
        V_repulsive2 = q.unsqueeze(1) * q_alpha.unsqueeze(0) * norms_rep2
        L_V_repulsive2 = V_repulsive2.sum(dim=0)  # size number of objects

        L_V_repulsive2 = L_V_repulsive2.view(-1)
        L_V_attractive_2 = L_V_attractive_2.view(-1)

        # if not tracking:
        #     #! add to terms function (divide by total number of showers per event)
        #     # L_V_repulsive = scatter_add(L_V_repulsive, object_index) / n_objects
        #     per_shower_weight = torch.exp(1 / (e_particles_pred_per_object + 0.4))
        #     soft_m = torch.nn.Softmax(dim=0)
        #     per_shower_weight = soft_m(per_shower_weight) * len(L_V_repulsive)
        #     L_V_repulsive = torch.mean(L_V_repulsive * per_shower_weight)
        # else:
        # if tracking:
        #     L_V_repulsive = torch.mean(L_V_repulsive * per_shower_weight)
        # else:
        if loss_type == "vrepweighted":
            L_V_repulsive = torch.sum(
                modified_showers.view(-1) * L_V_repulsive.view(-1)
            ) / len(modified_showers)
            L_V_repulsive2 = torch.sum(
                modified_showers.view(-1) * L_V_repulsive2.view(-1)
            ) / len(modified_showers)
        else:
            L_V_repulsive = torch.mean(L_V_repulsive)
            L_V_repulsive2 = torch.mean(L_V_repulsive2)
    else:
        L_V_repulsive = (
            scatter_add(V_repulsive.sum(dim=0), batch_object)
            / (n_hits_per_event * nope)
        ).sum()

    L_V = (
        attr_weight * L_V_attractive + repul_weight * L_V_repulsive
    )
    n_noise_hits_per_event = scatter_count(batch[is_noise])
    n_noise_hits_per_event[n_noise_hits_per_event == 0] = 1
    L_beta_noise = (
        s_B
        * (
            (scatter_add(beta[is_noise], batch[is_noise])) / n_noise_hits_per_event
        ).sum()
    )
    if loss_type == "hgcalimplementation":
        beta_per_object_c = scatter_add(beta[is_sig], object_index)
        beta_alpha = beta[is_sig][index_alpha]
        L_beta_sig = torch.mean(
            1 - beta_alpha + 1 - torch.clip(beta_per_object_c, 0, 1)
        )

        L_beta_noise = L_beta_noise / 4
        # ? note: the training that worked quite well was dividing this by the batch size (1/4)

    elif loss_type == "vrepweighted":
        # version one:
        beta_per_object_c = scatter_add(beta[is_sig], object_index)
        beta_alpha = beta[is_sig][index_alpha]
        L_beta_sig = 1 - beta_alpha + 1 - torch.clip(beta_per_object_c, 0, 1)
        L_beta_sig = torch.sum(L_beta_sig.view(-1) * modified_showers.view(-1))
        L_beta_sig = L_beta_sig / len(modified_showers)

        L_beta_noise = L_beta_noise / batch_size
        # ? note: the training that worked quite well was dividing this by the batch size (1/4)

    elif beta_term_option == "paper":
        beta_alpha = beta[is_sig][index_alpha]
        L_beta_sig = torch.sum(  # maybe 0.5 for less aggressive loss
            scatter_add((1 - beta_alpha), batch_object) / n_objects_per_event
        )
        # print("L_beta_sig", L_beta_sig / batch_size)
        # beta_exp = beta[is_sig]
        # beta_exp[index_alpha] = 0
        # # L_exp = torch.mean(beta_exp)
        # beta_exp = torch.exp(0.5 * beta_exp)
        # L_exp = torch.mean(scatter_add(beta_exp, batch) / n_hits_per_event)

    elif beta_term_option == "short-range-potential":
        # First collect the norms: We only want norms of hits w.r.t. the object they
        # belong to (like in V_attractive)
        # Apply transformation first, and then apply mask to keep only the norms we want,
        # then sum over hits, so the result is (n_objects,)
        norms_beta_sig = (1.0 / (20.0 * norms[is_sig] ** 2 + 1.0) * M[is_sig]).sum(
            dim=0
        )
        assert torch.all(norms_beta_sig >= 1.0) and torch.all(
            norms_beta_sig <= n_hits_per_object
        )
        # Subtract from 1. to remove self interaction, divide by number of hits per object
        norms_beta_sig = (1.0 - norms_beta_sig) / n_hits_per_object
        assert torch.all(norms_beta_sig >= -1.0) and torch.all(norms_beta_sig <= 0.0)
        norms_beta_sig *= beta_alpha
        # Conclusion:
        # lower beta --> higher loss (less negative)
        # higher norms --> higher loss

        # Sum over objects, divide by number of objects per event, then sum over events
        L_beta_norms_term = (
            scatter_add(norms_beta_sig, batch_object) / n_objects_per_event
        ).sum()
        assert L_beta_norms_term >= -batch_size and L_beta_norms_term <= 0.0

        # Logbeta term: Take -.2*torch.log(beta_alpha[is_object]+1e-9), sum it over objects,
        # divide by n_objects_per_event, then sum over events (same pattern as above)
        # lower beta --> higher loss
        L_beta_logbeta_term = (
            scatter_add(-0.2 * torch.log(beta_alpha + 1e-9), batch_object)
            / n_objects_per_event
        ).sum()

        # Final L_beta term
        L_beta_sig = L_beta_norms_term + L_beta_logbeta_term

    else:
        valid_options = ["paper", "short-range-potential"]
        raise ValueError(
            f'beta_term_option "{beta_term_option}" is not valid, choose from {valid_options}'
        )

    L_beta = L_beta_noise + L_beta_sig
    if beta_type == "pt" or beta_type == "pt+bc":
        L_beta = torch.zeros_like(L_beta)

    #L_alpha_coordinates = torch.mean(torch.norm(x_alpha_original - x_alpha, p=2, dim=1))
    x_original = original_coords / torch.norm(original_coords, p=2, dim=1).view(-1, 1)
    x_virtual  = cluster_space_coords / torch.norm(cluster_space_coords, p=2, dim=1).view(-1, 1)
    loss_coord = torch.mean(torch.norm(x_original - x_virtual, p=2, dim=1)) # We just compare the direction
    if beta_type == "pt+bc":
        assert noise_logits is not None
        y_true_noise = 1 - is_noise.float()
        num_positives = torch.sum(y_true_noise).item()
        num_negatives = len(y_true_noise) - num_positives
        num_all = len(y_true_noise)
        # Compute weights
        pos_weight = num_all / num_positives if num_positives > 0 else 0
        neg_weight = num_all / num_negatives if num_negatives > 0 else 0
        weight = pos_weight * y_true_noise + neg_weight * (1 - y_true_noise)
        L_bc = torch.nn.BCELoss(weight=weight)(
            noise_logits, 1-is_noise.float()
        )

    if torch.isnan(L_beta / batch_size):
        print("isnan!!!")
        print(L_beta, batch_size)
        print("L_beta_noise", L_beta_noise)
        print("L_beta_sig", L_beta_sig)
    result = {
        "loss_potential": L_V,  # 0
        "loss_beta": L_beta,
        "loss_beta_sig": L_beta_sig, # signal part of the betas
        "loss_beta_noise": L_beta_noise, # noise part of the betas
        "loss_attractive": L_V_attractive,
        "loss_repulsive": L_V_repulsive,
        "loss_coord": loss_coord,
    }
    if beta_type == "pt+bc":
        result["loss_noise_classification"] = L_bc
    return result



def huber(d, delta):
    """
    See: https://en.wikipedia.org/wiki/Huber_loss#Definition
    Multiplied by 2 w.r.t Wikipedia version (aligning with Jan's definition)
    """
    return torch.where(
        torch.abs(d) <= delta, d**2, 2.0 * delta * (torch.abs(d) - delta)
    )


def batch_cluster_indices(
    cluster_id: torch.Tensor, batch: torch.Tensor
) -> Tuple[torch.LongTensor, torch.LongTensor]:
    """
    Turns cluster indices per event to an index in the whole batch
    Example:
    cluster_id = torch.LongTensor([0, 0, 1, 1, 2, 0, 0, 1, 1, 1, 0, 0, 1])
    batch = torch.LongTensor([0, 0, 0, 0, 0, 1, 1, 1, 1, 1, 2, 2, 2])
    -->
    offset = torch.LongTensor([0, 0, 0, 0, 0, 3, 3, 3, 3, 3, 5, 5, 5])
    output = torch.LongTensor([0, 0, 1, 1, 2, 3, 3, 4, 4, 4, 5, 5, 6])
    """
    device = cluster_id.device
    assert cluster_id.device == batch.device
    # Count the number of clusters per entry in the batch
    n_clusters_per_event = scatter_max(cluster_id, batch, dim=-1)[0] + 1
    # Offsets are then a cumulative sum
    offset_values_nozero = n_clusters_per_event[:-1].cumsum(dim=-1)
    # Prefix a zero
    offset_values = torch.cat((torch.zeros(1, device=device), offset_values_nozero))
    # Fill it per hit
    offset = torch.gather(offset_values, 0, batch).long()
    return offset + cluster_id, n_clusters_per_event


def get_clustering_np(
    betas: np.array, X: np.array, tbeta: float = 0.1, td: float = 1.0
) -> np.array:
    """
    Returns a clustering of hits -> cluster_index, based on the GravNet model
    output (predicted betas and cluster space coordinates) and the clustering
    parameters tbeta and td.
    Takes numpy arrays as input.
    """
    n_points = betas.shape[0]
    select_condpoints = betas > tbeta
    # Get indices passing the threshold
    indices_condpoints = np.nonzero(select_condpoints)[0]
    # Order them by decreasing beta value
    indices_condpoints = indices_condpoints[np.argsort(-betas[select_condpoints])]
    # Assign points to condensation points
    # Only assign previously unassigned points (no overwriting)
    # Points unassigned at the end are bkg (-1)
    unassigned = np.arange(n_points)
    clustering = -1 * np.ones(n_points, dtype=np.int32)
    for index_condpoint in indices_condpoints:
        d = np.linalg.norm(X[unassigned] - X[index_condpoint], axis=-1)
        assigned_to_this_condpoint = unassigned[d < td]
        clustering[assigned_to_this_condpoint] = index_condpoint
        unassigned = unassigned[~(d < td)]
    return clustering


def get_clustering(betas: torch.Tensor, X: torch.Tensor, tbeta=0.1, td=1.0):
    """
    Returns a clustering of hits -> cluster_index, based on the GravNet model
    output (predicted betas and cluster space coordinates) and the clustering
    parameters tbeta and td.
    Takes torch.Tensors as input.
    """
    n_points = betas.size(0)
    select_condpoints = betas > tbeta
    # Get indices passing the threshold
    indices_condpoints = select_condpoints.nonzero()
    # Order them by decreasing beta value
    indices_condpoints = indices_condpoints[(-betas[select_condpoints]).argsort()]
    # Assign points to condensation points
    # Only assign previously unassigned points (no overwriting)
    # Points unassigned at the end are bkg (-1)
    unassigned = torch.arange(n_points)
    clustering = -1 * torch.ones(n_points, dtype=torch.long)
    for index_condpoint in indices_condpoints:
        d = torch.norm(X[unassigned] - X[index_condpoint][0], dim=-1)
        assigned_to_this_condpoint = unassigned[d < td]
        clustering[assigned_to_this_condpoint] = index_condpoint[0]
        unassigned = unassigned[~(d < td)]
    return clustering


def scatter_count(input: torch.Tensor):
    """
    Returns ordered counts over an index array
    Example:
    >>> scatter_count(torch.Tensor([0, 0, 0, 1, 1, 2, 2])) # input
    >>> [3, 2, 2]
    Index assumptions work like in torch_scatter, so:
    >>> scatter_count(torch.Tensor([1, 1, 1, 2, 2, 4, 4]))
    >>> tensor([0, 3, 2, 0, 2])
    """
    return scatter_add(torch.ones_like(input, dtype=torch.long), input.long())


def scatter_counts_to_indices(input: torch.LongTensor) -> torch.LongTensor:
    """
    Converts counts to indices. This is the inverse operation of scatter_count
    Example:
    input:  [3, 2, 2]
    output: [0, 0, 0, 1, 1, 2, 2]
    """
    return torch.repeat_interleave(
        torch.arange(input.size(0), device=input.device), input
    ).long()


def get_inter_event_norms_mask(
    batch: torch.LongTensor, nclusters_per_event: torch.LongTensor
):
    """
    Creates mask of (nhits x nclusters) that is only 1 if hit i is in the same event as cluster j
    Example:
    cluster_id_per_event = torch.LongTensor([0, 0, 1, 1, 2, 0, 0, 1, 1, 1, 0, 0, 1])
    batch = torch.LongTensor([0, 0, 0, 0, 0, 1, 1, 1, 1, 1, 2, 2, 2])
    Should return:
    torch.LongTensor([
        [1, 1, 1, 0, 0, 0, 0],
        [1, 1, 1, 0, 0, 0, 0],
        [1, 1, 1, 0, 0, 0, 0],
        [1, 1, 1, 0, 0, 0, 0],
        [1, 1, 1, 0, 0, 0, 0],
        [0, 0, 0, 1, 1, 0, 0],
        [0, 0, 0, 1, 1, 0, 0],
        [0, 0, 0, 1, 1, 0, 0],
        [0, 0, 0, 1, 1, 0, 0],
        [0, 0, 0, 1, 1, 0, 0],
        [0, 0, 0, 0, 0, 1, 1],
        [0, 0, 0, 0, 0, 1, 1],
        [0, 0, 0, 0, 0, 1, 1],
        ])
    """
    device = batch.device
    # Following the example:
    # Expand batch to the following (nhits x nevents) matrix (little hacky, boolean mask -> long):
    # [[1, 1, 1, 1, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0],
    #  [0, 0, 0, 0, 0, 1, 1, 1, 1, 1, 1, 0, 0, 0],
    #  [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 1, 1]]
    batch_expanded_as_ones = (
        batch
        == torch.arange(batch.max() + 1, dtype=torch.long, device=device).unsqueeze(-1)
    ).long()
    # Then repeat_interleave it to expand it to nclusters rows, and transpose to get (nhits x nclusters)
    return batch_expanded_as_ones.repeat_interleave(nclusters_per_event, dim=0).T


def isin(ar1, ar2):
    """To be replaced by torch.isin for newer releases of torch"""
    return (ar1[..., None] == ar2).any(-1)


def reincrementalize(y: torch.Tensor, batch: torch.Tensor) -> torch.Tensor:
    """Re-indexes y so that missing clusters are no longer counted.
    Example:
        >>> y = torch.LongTensor([
            0, 0, 0, 1, 1, 3, 3,
            0, 0, 0, 0, 0, 2, 2, 3, 3,
            0, 0, 1, 1
            ])
        >>> batch = torch.LongTensor([
            0, 0, 0, 0, 0, 0, 0,
            1, 1, 1, 1, 1, 1, 1, 1, 1,
            2, 2, 2, 2,
            ])
        >>> print(reincrementalize(y, batch))
        tensor([0, 0, 0, 1, 1, 2, 2, 0, 0, 0, 0, 0, 1, 1, 2, 2, 0, 0, 1, 1])
    """
    y_offset, n_per_event = batch_cluster_indices(y, batch)
    offset = y_offset - y
    n_clusters = n_per_event.sum()
    holes = (
        (~isin(torch.arange(n_clusters, device=y.device), y_offset))
        .nonzero()
        .squeeze(-1)
    )
    n_per_event_without_holes = n_per_event.clone()
    n_per_event_cumsum = n_per_event.cumsum(0)
    for hole in holes.sort(descending=True).values:
        y_offset[y_offset > hole] -= 1
        i_event = (hole > n_per_event_cumsum).long().argmin()
        n_per_event_without_holes[i_event] -= 1
    offset_per_event = torch.zeros_like(n_per_event_without_holes)
    offset_per_event[1:] = n_per_event_without_holes.cumsum(0)[:-1]
    offset_without_holes = torch.gather(offset_per_event, 0, batch).long()
    reincrementalized = y_offset - offset_without_holes
    return reincrementalized


def L_clusters_calc(batch, cluster_space_coords, cluster_index, frac_combinations, q):
    number_of_pairs = 0
    for batch_id in batch.unique():
        # do all possible pairs...
        bmask = batch == batch_id
        clust_space_filt = cluster_space_coords[bmask]
        pos_pairs_all = []
        neg_pairs_all = []
        if len(cluster_index[bmask].unique()) <= 1:
            continue
        L_clusters = torch.tensor(0.0).to(q.device)
        for cluster in cluster_index[bmask].unique():
            coords_pos = clust_space_filt[cluster_index[bmask] == cluster]
            coords_neg = clust_space_filt[cluster_index[bmask] != cluster]
            if len(coords_neg) == 0:
                continue
            clust_idx = cluster_index[bmask] == cluster
            # all_ones = torch.ones_like((clust_idx, clust_idx))
            # pos_pairs = [[i, j] for i in range(len(coords_pos)) for j in range (len(coords_pos)) if i < j]
            total_num = (len(coords_pos) ** 2) / 2
            num = int(frac_combinations * total_num)
            pos_pairs = []
            for i in range(num):
                pos_pairs.append(
                    [
                        np.random.randint(len(coords_pos)),
                        np.random.randint(len(coords_pos)),
                    ]
                )
            neg_pairs = []
            for i in range(len(pos_pairs)):
                neg_pairs.append(
                    [
                        np.random.randint(len(coords_pos)),
                        np.random.randint(len(coords_neg)),
                    ]
                )
            pos_pairs_all += pos_pairs
            neg_pairs_all += neg_pairs
        pos_pairs = torch.tensor(pos_pairs_all)
        neg_pairs = torch.tensor(neg_pairs_all)
        """# do just a small sample of the pairs. ...
        bmask = batch == batch_id

        #L_clusters = 0   # Loss of randomly sampled distances between points inside and outside clusters

        pos_idx, neg_idx = [], []
        for cluster in cluster_index[bmask].unique():
            clust_idx = (cluster_index == cluster)[bmask]
            perm = torch.randperm(clust_idx.sum())
            perm1 = torch.randperm((~clust_idx).sum())
            perm2 = torch.randperm(clust_idx.sum())
            #cutoff = clust_idx.sum()//2
            pos_lst = clust_idx.nonzero()[perm]
            neg_lst = (~clust_idx).nonzero()[perm1]
            neg_lst_second = clust_idx.nonzero()[perm2]
            if len(pos_lst) % 2:
                pos_lst = pos_lst[:-1]
            if len(neg_lst) % 2:
                neg_lst = neg_lst[:-1]
            len_cap = min(len(pos_lst), len(neg_lst), len(neg_lst_second))
            if len_cap % 2:
                len_cap -= 1
            pos_lst = pos_lst[:len_cap]
            neg_lst = neg_lst[:len_cap]
            neg_lst_second = neg_lst_second[:len_cap]
            pos_pairs = pos_lst.reshape(-1, 2)
            neg_pairs = torch.cat([neg_lst, neg_lst_second], dim=1)
            neg_pairs = neg_pairs[:pos_lst.shape[0]//2, :]
            pos_idx.append(pos_pairs)
            neg_idx.append(neg_pairs)
        pos_idx = torch.cat(pos_idx)
        neg_idx = torch.cat(neg_idx)"""
        assert pos_pairs.shape == neg_pairs.shape
        if len(pos_pairs) == 0:
            continue
        cluster_space_coords_filtered = cluster_space_coords[bmask]
        qs_filtered = q[bmask]
        pos_norms = (
            cluster_space_coords_filtered[pos_pairs[:, 0]]
            - cluster_space_coords_filtered[pos_pairs[:, 1]]
        ).norm(dim=-1)

        neg_norms = (
            cluster_space_coords_filtered[neg_pairs[:, 0]]
            - cluster_space_coords_filtered[neg_pairs[:, 1]]
        ).norm(dim=-1)
        q_pos = qs_filtered[pos_pairs[:, 0]]
        q_neg = qs_filtered[neg_pairs[:, 0]]
        q_s = torch.cat([q_pos, q_neg])
        norms_pos = torch.cat([pos_norms, neg_norms])
        ys = torch.cat([torch.ones_like(pos_norms), -torch.ones_like(neg_norms)])
        L_clusters += torch.sum(
            q_s * torch.nn.HingeEmbeddingLoss(reduce=None)(norms_pos, ys)
        )
        number_of_pairs += norms_pos.shape[0]
    if number_of_pairs > 0:
        L_clusters = L_clusters / number_of_pairs

    return L_clusters

def calc_eta_phi(coords):
    """
    Calculate eta and phi from cartesian coordinates
    """
    x = coords[:, 0]
    y = coords[:, 1]
    z = coords[:, 2]
    #eta, phi = torch.atan2(y, x), torch.asin(z / coords.norm(dim=1))
    phi = torch.arctan2(y, x)
    eta = torch.arctanh(z / torch.sqrt(x**2 + y**2 + z**2))
    return torch.stack([eta, phi], dim=1)

def object_condensation_loss(
        batch, # input event
        pred,
        labels,
        batch_numbers,
        q_min=3.0,
        frac_clustering_loss=0.1,
        attr_weight=1.0,
        repul_weight=1.0,
        fill_loss_weight=1.0,
        use_average_cc_pos=0.0,
        loss_type="hgcalimplementation",
        clust_space_norm="none",
        dis=False,
        coord_weight=0.0,
        beta_type="default",
        lorentz_norm=False
):
    """
    :param batch: Model input
    :param pred: Model output, containing regressed coordinates + betas
    :param clust_space_dim: Number of dimensions in the cluster space
    :return:
    """
    _, S = pred.shape
    noise_logits = None
    if beta_type == "default":
        clust_space_dim = S - 1
        bj = torch.sigmoid(torch.reshape(pred[:, clust_space_dim], [-1, 1])) # betas
    elif beta_type == "pt":
        bj = batch.pt
        clust_space_dim = S
    elif beta_type == "pt+bc":
        bj = batch.pt
        clust_space_dim = S - 1
        noise_logits = pred[:, clust_space_dim]
    original_coords = batch.input_vectors
    if dis:
        distance_threshold = torch.reshape(pred[:, -1], [-1, 1])
    else:
        distance_threshold = 0
    xj = pred[:, :clust_space_dim] # Coordinates in clustering space
    #xj = calc_eta_phi(xj)
    if clust_space_norm == "twonorm":
        xj = torch.nn.functional.normalize(xj, dim=1)
    elif clust_space_norm == "tanh":
        xj = torch.tanh(xj)
    elif clust_space_norm == "none":
        pass
    else:
        raise NotImplementedError

    clustering_index_l = labels

    a = calc_LV_Lbeta(
        original_coords,
        batch,
        distance_threshold,
        beta=bj.view(-1),
        cluster_space_coords=xj,  # Predicted by model
        cluster_index_per_event=clustering_index_l.view(
            -1
        ).long(),  # Truth hit->cluster index
        batch=batch_numbers.long(),
        qmin=q_min,
        attr_weight=attr_weight,
        repul_weight=repul_weight,
        use_average_cc_pos=use_average_cc_pos,
        loss_type=loss_type,
        dis=dis,
        beta_type=beta_type,
        noise_logits=noise_logits,
        lorentz_norm=lorentz_norm
    )

    loss = a["loss_potential"] + a["loss_beta"]
    if coord_weight > 0:
        loss += a["loss_coord"] * coord_weight
    if beta_type == "pt+bc":
        loss += a["loss_noise_classification"]
    return loss, a
