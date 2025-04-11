import numpy as np
import torch
#from torch_scatter import scatter_add, scatter_sum, scatter_mean

from src.dataset.functions_data import (
    get_ratios,
    find_mask_no_energy,
    find_cluster_id,
    get_particle_features,
    get_hit_features,
    calculate_distance_to_boundary,
    concatenate_Particles_GT,
    create_noise_label,
    EventJets,
    EventPFCands,
    EventCollection,
    Event,
    EventMetadataAndMET,
    concat_event_collection
)


def create_inputs_from_table(
    output, hits_only, prediction=False, hit_chis=False, pos_pxpy=False, is_Ks=False
):
    """Used by graph creation to get nodes and edge features

    Args:
        output (_type_): input from the root reading
        hits_only (_type_): reading only hits or also tracks
        prediction (bool, optional): if running in eval mode. Defaults to False.

    Returns:
        _type_: all information to construct a graph
    """
    graph_empty = False
    number_hits = np.int32(np.sum(output["pf_mask"][0]))
    number_part = np.int32(np.sum(output["pf_mask"][1]))
    
    (
        pos_xyz_hits,
        pos_pxpypz,
        p_hits,
        e_hits,
        hit_particle_link,
        pandora_cluster,
        pandora_cluster_energy,
        pfo_energy,
        pandora_mom,
        pandora_ref_point,
        pandora_pid, 
        unique_list_particles,
        cluster_id,
        hit_type_feature,
        pandora_pfo_link,
        daughters,
        hit_link_modified,
        connection_list,
        chi_squared_tracks,
    ) = get_hit_features(
        output,
        number_hits,
        prediction,
        number_part,
        hit_chis=hit_chis,
        pos_pxpy=pos_pxpy,
        is_Ks=is_Ks,
    )
    # features particles
    if torch.sum(torch.Tensor(unique_list_particles)>20000)>0:
        graph_empty = True
    else:
        y_data_graph = get_particle_features(
            unique_list_particles, output, prediction, connection_list
        )
        assert len(y_data_graph) == len(unique_list_particles)
        # remove particles that have no energy, no hits or only track hits
        if not is_Ks:
            mask_hits, mask_particles = find_mask_no_energy(
                cluster_id,
                hit_type_feature,
                e_hits,
                y_data_graph,
                daughters,
                prediction,
                is_Ks=is_Ks,
            )
            # create mapping from links to number of particles in the event
            cluster_id, unique_list_particles = find_cluster_id(hit_particle_link[~mask_hits])
            y_data_graph.mask(~mask_particles)
        else:
            mask_hits = torch.zeros_like(e_hits).bool().view(-1)
        if prediction:
            if is_Ks:
                result = [
                    y_data_graph,  # y_data_graph[~mask_particles],
                    p_hits[~mask_hits],
                    e_hits[~mask_hits],
                    cluster_id,
                    hit_particle_link[~mask_hits],
                    pos_xyz_hits[~mask_hits],
                    pos_pxpypz[~mask_hits],
                    pandora_cluster[~mask_hits],
                    pandora_cluster_energy[~mask_hits],
                    pandora_mom[~mask_hits],
                    pandora_ref_point[~mask_hits],
                    pandora_pid[~mask_hits],
                    pfo_energy[~mask_hits],
                    pandora_pfo_link[~mask_hits],
                    hit_type_feature[~mask_hits],
                    hit_link_modified[~mask_hits],
                    daughters[~mask_hits],
                ]
            else:
                result = [
                    y_data_graph,  # y_data_graph[~mask_particles],
                    p_hits[~mask_hits],
                    e_hits[~mask_hits],
                    cluster_id,
                    hit_particle_link[~mask_hits],
                    pos_xyz_hits[~mask_hits],
                    pos_pxpypz[~mask_hits],
                    pandora_cluster[~mask_hits],
                    pandora_cluster_energy[~mask_hits],
                    pandora_mom,
                    pandora_ref_point,
                    pandora_pid, 
                    pfo_energy[~mask_hits],
                    pandora_pfo_link[~mask_hits],
                    hit_type_feature[~mask_hits],
                    hit_link_modified[~mask_hits],
                ]
        else:
            result = [
                y_data_graph,  # y_data_graph[~mask_particles],
                p_hits[~mask_hits],
                e_hits[~mask_hits],
                cluster_id,
                hit_particle_link[~mask_hits],
                pos_xyz_hits[~mask_hits],
                pos_pxpypz[~mask_hits],
                pandora_cluster,
                pandora_cluster_energy,
                pandora_mom,
                pandora_ref_point,
                pandora_pid, 
                pfo_energy,
                pandora_pfo_link,
                hit_type_feature[~mask_hits],
                hit_link_modified[~mask_hits],
            ]
        if hit_chis:
            result.append(
                chi_squared_tracks[~mask_hits],
            )
        else:
            result.append(None)
        hit_type = hit_type_feature[~mask_hits]
        # if hits only remove tracks, otherwise leave tracks
        if hits_only:
            hit_mask = (hit_type == 0) | (hit_type == 1)
            hit_mask = ~hit_mask
            for i in range(1, len(result)):
                if result[i] is not None:
                    result[i] = result[i][hit_mask]
            hit_type_one_hot = torch.nn.functional.one_hot(
                hit_type_feature[~mask_hits][hit_mask] - 2, num_classes=2
            )

        else:
            # if we want the tracks keep only 1 track hit per charged particle.
            hit_mask = hit_type == 10
            hit_mask = ~hit_mask
            for i in range(1, len(result)):
                if result[i] is not None:
                    # if len(result[i].shape) == 2 and result[i].shape[0] == 3:
                    #     result[i] = result[i][:, hit_mask]
                    # else:
                    #     result[i] = result[i][hit_mask]
                    result[i] = result[i][hit_mask]
            hit_type_one_hot = torch.nn.functional.one_hot(
                hit_type_feature[~mask_hits][hit_mask], num_classes=5
            )
        result.append(hit_type_one_hot)
        result.append(connection_list)
        return result
    if graph_empty:
        return [None]

def remove_hittype0(graph):
    filt = graph.ndata["hit_type"] == 0
    # graph.ndata["hit_type"] -= 1
    return dgl.remove_nodes(graph, torch.where(filt)[0])

def store_track_at_vertex_at_track_at_calo(graph):
    # To make it compatible with clustering, remove the 0 hit type nodes and store them as pos_pxpypz_at_vertex
    tracks_at_calo = graph.ndata["hit_type"] == 1
    tracks_at_vertex = graph.ndata["hit_type"] == 0
    part = graph.ndata["particle_number"].long()
    assert (part[tracks_at_calo] == part[tracks_at_vertex]).all()
    graph.ndata["pos_pxpypz_at_vertex"] = torch.zeros_like(graph.ndata["pos_pxpypz"])
    graph.ndata["pos_pxpypz_at_vertex"][tracks_at_calo] = graph.ndata["pos_pxpypz"][tracks_at_vertex]
    return remove_hittype0(graph)

def create_jets_outputs_Delphes(output):
    n_ch = int(output["n_CH"][0, 0])
    n_nh = int(output["n_NH"][0, 0])
    n_photons = int(output["n_photon"][0, 0])
    n_genp = int(output["NParticles"][0, 0])
    print(type(output["CH"]), output["CH"].shape, output["CH"])
    ch = output["CH"][:, :n_ch]
    nh = output["NH"][:, :n_nh]
    photons = output["EFlowPhoton"][:, :n_photons]
    genp = output["GenParticles"][:, :n_genp]
    nh_mass = [0.939] * n_nh
    nh_charge = [0] * n_nh
    nh_pid = [2112] * n_nh
    nh_jets = [-1] * n_nh
    ch_charge = ch[4, :]
    ch_pid = [211] * n_ch
    ch_jets = [-1] * n_ch
    photons_jets = [-1] * n_photons
    photons_mass = [0] * n_photons
    photons_charge = [0] * n_photons
    photons_pid = [22] * n_photons
    nh = nh.T
    ch = ch.T
    photons = photons.T
    genp = genp.T
    nh_data = EventPFCands(nh[:, 2], nh[:, 0], nh[:, 1], nh_mass, nh_charge, nh_pid, pf_cand_jet_idx=nh_jets)
    ch_data = EventPFCands(ch[:, 2], ch[:, 0], ch[:, 1], ch[:, 3], ch_charge, ch_pid, pf_cand_jet_idx=ch_jets)
    photon_data = EventPFCands(photons[:, 2], photons[:, 0], photons[:, 1], photons_mass, photons_charge,
                               photons_pid, pf_cand_jet_idx=photons_jets)
    pfcands = concat_event_collection([nh_data, ch_data, photon_data], nobatch=1)
    filter_pfcands = (pfcands.pt > 0.5) & (torch.abs(pfcands.eta) < 2.4)
    pfcands.mask(filter_pfcands)
    genp_status = genp[:, 6]
    genp_eta = genp[:, 0]
    genp_pt = genp[:, 2]
    filter_dq = genp_status == 23
    filter_partons = (genp_status >= 51) & (genp_status <= 59) & (np.abs(genp_eta) < 2.4) & (genp_pt > 0.5)
    matrix_element_gen_particles = EventPFCands(
        genp[filter_dq, 2],
        genp[filter_dq, 0],
        genp[filter_dq, 1],
        genp[filter_dq, 3],
        np.sign(genp[filter_dq, 4]),
        genp[filter_dq, 5],
        pf_cand_jet_idx=-1 * np.ones_like(genp[filter_dq, 0]),
    )
    parton_level_particles = EventPFCands(
        genp[filter_partons, 2],
        genp[filter_partons, 0],
        genp[filter_partons, 1],
        genp[filter_partons, 3],
        np.sign(genp[filter_partons, 4]),
        genp[filter_partons, 5],
        pf_cand_jet_idx=-1 * np.ones_like(genp[filter_partons, 0]),
    )
    filter_final_gen_particles = (genp_status == 1) & (np.abs(genp_eta) < 2.4) & (genp_pt > 0.5)
    final_gen_particles = EventPFCands(
        genp[filter_final_gen_particles, 2],
        genp[filter_final_gen_particles, 0],
        genp[filter_final_gen_particles, 1],
        genp[filter_final_gen_particles, 3],
        np.sign(genp[filter_final_gen_particles, 4]),
        genp[filter_final_gen_particles, 5],
        pf_cand_jet_idx=-1 * np.ones_like(genp[filter_final_gen_particles, 0]),
    )
    if len(final_gen_particles) == 0:
        print("No gen particles in this event?")
        print(genp_status, len(genp_status))
        #print(genp_eta)

    return Event(pfcands=pfcands, matrix_element_gen_particles=matrix_element_gen_particles,
                 final_gen_particles=final_gen_particles, final_parton_level_particles=parton_level_particles)

def create_jets_outputs(
    output,
    config=None,
):
    n_jets = int(output["n_jets"][0, 0])
    jets_data = output["jets"][:, :n_jets]
    n_genjets = int(output["n_genjets"][0, 0])
    genjets_data = output["genjets"][:, :n_genjets]
    n_pfcands = int(output["n_pfcands"][0, 0])
    n_fat_jets = int(output["n_fat_jets"][0, 0])
    fat_jets_data = output["fat_jets"][:, :n_fat_jets]
    #jets_data = EventJets(jets_data[:, 0], )
    return jets_data, genjets_data, fat_jets_data

def create_jets_outputs_new(
    output, separate_special_pfcands=False
):
    print(output)
    n_jets = int(output["n_jets"][0, 0])
    jets_data = output["jets"][:, :n_jets]
    n_genjets = int(output["n_genjets"][0, 0])
    genjets_data = output["genjets"][:, :n_genjets]
    n_pfcands = int(output["n_pfcands"][0, 0])
    pfcands_data = output["pfcands"][:, :n_pfcands]
    pfcands_jets_mapping = output["pfcands_jet_mapping"]
    output_MET = output["MET"]
    n_fat_jets = int(output["n_fat_jets"][0, 0])
    fat_jets_data = output["fat_jets"][:, :n_fat_jets]
    num_mapping = np.argmax(pfcands_jets_mapping[1]) + 1
    if n_jets == 0:
        num_mapping = 0

    n_electrons = int(output["n_electrons"][0, 0])
    electrons_data = output["electrons"][:, :n_electrons]
    n_muons = int(output["n_muons"][0, 0])
    muons_data = output["muons"][:, :n_muons]
    n_photons = int(output["n_photons"][0, 0])
    photons_data = output["photons"][:, :n_photons]
    matrix_element_gen_particles_data = output["matrix_element_gen_particles"]
    if "final_gen_particles" in output:
        # new config
        #n_final_gen_particles = int(output["n_final_gen_particles"][0, 0])
        final_gen_particles_data = output["final_gen_particles"]#[:, :n_final_gen_particles]
        final_parton_level_particles_data = output["final_parton_level_particles"]#[:, :n_final_gen_particles]

    pfcands_jets_mapping = pfcands_jets_mapping[:, :num_mapping]
    #n_offline_pfcands = int(output["n_offline_pfcands"][0, 0])
    #offline_pfcands_data = output["offline_pfcands"][:, :n_offline_pfcands]
    #offline_jets_mapping = output["offline_pfcands_jet_mapping"]
    #num_mapping_offline = np.argmax(offline_jets_mapping[1]) + 1
    #assert offline_jets_mapping[1].max() < n_offline_pfcands
    if len(pfcands_jets_mapping[1]):
        assert pfcands_jets_mapping[1].max() < n_pfcands
    #offline_jets_mapping = offline_jets_mapping[:, :num_mapping_offline]
    jets_data = jets_data.T
    genjets_data = genjets_data.T
    pfcands_data = pfcands_data.T
    fat_jets_data = fat_jets_data.T
    matrix_element_gen_particles_data = matrix_element_gen_particles_data.T
    matrix_element_gen_particles_data = EventPFCands(pt=matrix_element_gen_particles_data[:, 0],
                                                     eta=matrix_element_gen_particles_data[:, 1],
                                                     phi=matrix_element_gen_particles_data[:, 2],
                                                     mass=matrix_element_gen_particles_data[:, 3],
                                                     charge=np.sign(matrix_element_gen_particles_data[:, 4]),
                                                     pid=matrix_element_gen_particles_data[:, 4],
                                                     pf_cand_jet_idx=-1*np.ones_like(matrix_element_gen_particles_data[:, 0]))
    if "final_gen_particles" in output:
        final_gen_particles_data = final_gen_particles_data.T
        final_parton_level_particles_data = final_parton_level_particles_data.T
        n_fp = torch.argmin(torch.tensor(final_gen_particles_data[:, 0])).item()
        n_pp = torch.argmin(torch.tensor(final_parton_level_particles_data[:, 0])).item()
        final_gen_particles_data = EventPFCands(pt=final_gen_particles_data[:n_fp, 0],
                                                eta=final_gen_particles_data[:n_fp, 1],
                                                phi=final_gen_particles_data[:n_fp, 2],
                                                mass=final_gen_particles_data[:n_fp, 3],
                                                charge=np.sign(final_gen_particles_data[:n_fp, 4]),
                                                pid=final_gen_particles_data[:n_fp, 4],
                                                pf_cand_jet_idx=-1*np.ones_like(final_gen_particles_data[:n_fp, 0]))
        final_parton_level_particles_data = EventPFCands(pt=final_parton_level_particles_data[:n_pp, 0],
                                                        eta=final_parton_level_particles_data[:n_pp, 1],
                                                        phi=final_parton_level_particles_data[:n_pp, 2],
                                                        mass=final_parton_level_particles_data[:n_pp, 3],
                                                        charge=np.sign(final_parton_level_particles_data[:n_pp, 4]),
                                                        pid=final_parton_level_particles_data[:n_pp, 4],
                                                        pf_cand_jet_idx=-1*np.ones_like(final_parton_level_particles_data[:n_pp, 0]),
                                                        status=final_parton_level_particles_data[:n_pp, 5])
    #offline_pfcands_data = offline_pfcands_data.T
    electrons_data = electrons_data.T
    muons_data = muons_data.T
    photons_data = photons_data.T
    electrons_mass = np.ones_like(electrons_data[:, 0]) * 0.511
    muons_mass = np.ones_like(muons_data[:, 0]) * 105.7
    photons_mass = np.zeros_like(photons_data[:, 0])
    electrons_pid = np.ones_like(electrons_data[:, 0]) * 0
    muons_pid = np.ones_like(muons_data[:, 0]) * 1
    photons_pid = np.ones_like(photons_data[:, 0]) * 2
    photons_charge = np.zeros_like(photons_data[:, 0])
    electrons_data = np.column_stack((electrons_data[:, 0], electrons_data[:, 1], electrons_data[:, 2],
                                      electrons_mass, electrons_data[:, 3], electrons_pid))
    muons_data = np.column_stack((muons_data[:, 0], muons_data[:, 1], muons_data[:, 2],
                                    muons_mass, muons_data[:, 3], muons_pid))
    photons_data = np.column_stack((photons_data[:, 0], photons_data[:, 1], photons_data[:, 2],
                                    photons_mass, photons_charge, photons_pid))
    special_pfcands_data = np.concatenate((electrons_data, muons_data, photons_data), axis=0)
    special_pfcands_data = torch.tensor(special_pfcands_data)
    # is there
    jets_data = EventJets(
        jets_data[:, 0],
        jets_data[:, 1],
        jets_data[:, 2],
        jets_data[:, 3],
        #jets_data[:, 4]
    )
    genjets_data = EventJets(
        genjets_data[:, 0],
        genjets_data[:, 1],
        genjets_data[:, 2],
        genjets_data[:, 3],
    )
    fatjets_data = EventJets(
        fat_jets_data[:, 0],
        fat_jets_data[:, 1],
        fat_jets_data[:, 2],
        fat_jets_data[:, 3],
        #fat_jets_data[:, 4]
    )
    pfcands_jets_mapping = list(pfcands_jets_mapping)
    #offline_jets_mapping = list(offline_jets_mapping)
    pfcands_data = EventPFCands(*[pfcands_data[:, i] for i in range(6)] + pfcands_jets_mapping)
    special_pfcands_data = EventPFCands(*[special_pfcands_data[:, i] for i in range(6)], pf_cand_jet_idx=-1*torch.ones_like(special_pfcands_data[:, 0]))
    if not separate_special_pfcands:
        pfcands_data = concat_event_collection([pfcands_data, special_pfcands_data])
        special_pfcands_data = None
    MET_data = EventMetadataAndMET(pt=output_MET[0], phi=output_MET[1], scouting_trig=output_MET[2], offline_trig=output_MET[3], veto_trig=output_MET[4])
    #offline_pfcands_data = EventPFCands(*[offline_pfcands_data[:, i] for i in range(6)] + offline_jets_mapping, offline=True)
    kwargs = {}
    if "final_gen_particles" in output:
        kwargs["final_gen_particles"] = final_gen_particles_data
        kwargs["final_parton_level_particles"] = final_parton_level_particles_data
    return Event(jets=jets_data, genjets=genjets_data, pfcands=pfcands_data, MET=MET_data, fatjets=fatjets_data,
                 matrix_element_gen_particles=matrix_element_gen_particles_data, special_pfcands=special_pfcands_data,
                 **kwargs)
    #return {
    #    "jets": jets_data,
    #    "genjets": genjets_data,
    #    "pfcands": pfcands_data,
    #    # "offline_pfcands": offline_pfcands_data
    #}

def create_graph(
    output,
    config=None,
    n_noise=0,
):
    graph_empty = False
    hits_only = config.graph_config.get(
        "only_hits", False
    )  # Whether to only include hits in the graph
    # standardize_coords = config.graph_config.get("standardize_coords", False)
    extended_coords = config.graph_config.get("extended_coords", False)
    prediction = config.graph_config.get("prediction", False)
    hit_chis = config.graph_config.get("hit_chis_track", False)
    pos_pxpy = config.graph_config.get("pos_pxpy", False)
    is_Ks = config.graph_config.get("ks", False)
    noise_class = config.graph_config.get("noise", False)
    result = create_inputs_from_table(
        output,
        hits_only=hits_only,
        prediction=prediction,
        hit_chis=hit_chis,
        pos_pxpy=pos_pxpy,
        is_Ks=is_Ks,
    )
    if len(result) == 1:
        graph_empty = True
        g = 0
        y_data_graph = 0
    else:
        (
            y_data_graph,
            p_hits,
            e_hits,
            cluster_id,
            hit_particle_link,
            pos_xyz_hits,
            pos_pxpypz,
            pandora_cluster,
            pandora_cluster_energy,
            pandora_mom,
            pandora_ref_point,
            pandora_pid, 
            pandora_pfo_energy,
            pandora_pfo_link,
            hit_type,
            hit_link_modified,
            daughters, 
            chi_squared_tracks,
            hit_type_one_hot,
            connections_list
        ) = result
        if noise_class:
            mask_loopers, mask_particles = create_noise_label(
            e_hits, hit_particle_link, y_data_graph, cluster_id
            )
            hit_particle_link[mask_loopers] = -1
            y_data_graph.mask(mask_particles)
            cluster_id, unique_list_particles = find_cluster_id(hit_particle_link)
        graph_coordinates = pos_xyz_hits  # / 3330  # divide by detector size
        graph_empty = False
        g = dgl.graph(([], []))
        g.add_nodes(graph_coordinates.shape[0])
        if hits_only == False:
            hit_features_graph = torch.cat(
                (graph_coordinates, hit_type_one_hot, e_hits, p_hits), dim=1
            )  # dims = 8
        else:
            hit_features_graph = torch.cat(
                (graph_coordinates, hit_type_one_hot, e_hits, p_hits), dim=1
            )  # dims = 9

        g.ndata["h"] = hit_features_graph
        g.ndata["pos_hits_xyz"] = pos_xyz_hits
        g.ndata["pos_pxpypz"] = pos_pxpypz
        g = calculate_distance_to_boundary(g)
        g.ndata["hit_type"] = hit_type
        g.ndata[
            "e_hits"
        ] = e_hits  # if no tracks this is e and if there are tracks this fills the tracks e values with p
        if hit_chis:
            g.ndata["chi_squared_tracks"] = chi_squared_tracks
        g.ndata["particle_number"] = cluster_id
        g.ndata["hit_link_modified"] = hit_link_modified
        g.ndata["particle_number_nomap"] = hit_particle_link
        if prediction:
            g.ndata["pandora_cluster"] = pandora_cluster
            g.ndata["pandora_pfo"] = pandora_pfo_link
            g.ndata["pandora_cluster_energy"] = pandora_cluster_energy
            g.ndata["pandora_pfo_energy"] = pandora_pfo_energy
            if is_Ks:
                g.ndata["pandora_momentum"] = pandora_mom
                g.ndata["pandora_reference_point"] = pandora_ref_point
                g.ndata["daughters"] = daughters
                g.ndata["pandora_pid"] = pandora_pid
        y_data_graph.calculate_corrected_E(g, connections_list)
        # if is_Ks == True:
        #     if y_data_graph.pid.flatten().shape[0] == 4 and np.count_nonzero(y_data_graph.pid.flatten() == 22) == 4:
        #         graph_empty = False
        #     else:
        #         graph_empty = True
        #     if g.ndata["h"].shape[0] < 10 or (set(g.ndata["hit_type"].unique().tolist()) == set([0, 1]) and g.ndata["hit_type"][g.ndata["hit_type"] == 1].shape[0] < 10):
        #         graph_empty = True  # less than 10 hits
        # print("y len", len(y_data_graph))
        # if is_Ks == False:
        #     if len(y_data_graph) < 4:
        #         graph_empty = True

        if pos_xyz_hits.shape[0] < 10:
            graph_empty = True
    if graph_empty:
        return [g, y_data_graph], graph_empty
    # print("graph_empty",graph_empty)
    g = store_track_at_vertex_at_track_at_calo(g)
    if noise_class:
        g = make_bad_tracks_noise_tracks(g)
    return [g, y_data_graph], graph_empty


def graph_batch_func(list_graphs):
    """collator function for graph dataloader

    Args:
        list_graphs (list): list of graphs from the iterable dataset

    Returns:
        batch dgl: dgl batch of graphs
    """
    list_graphs_g = [el[0] for el in list_graphs]
    # list_y = add_batch_number(list_graphs)
    # ys = torch.cat(list_y, dim=0)
    # ys = torch.reshape(ys, [-1, list_y[0].shape[1]])
    ys = concatenate_Particles_GT(list_graphs)
    bg = dgl.batch(list_graphs_g)
    # reindex particle number
    return bg, ys

def make_bad_tracks_noise_tracks(g):
    # is_chardged =scatter_add((g.ndata["hit_type"]==1).view(-1), g.ndata["particle_number"].long())[1:]
    mask_hit_type_t1 = g.ndata["hit_type"]==2
    mask_hit_type_t2 = g.ndata["hit_type"]==1
    mask_all = mask_hit_type_t1
    # the other error could come from no hits in the ECAL for a cluster
    mean_pos_cluster = scatter_mean(g.ndata["pos_hits_xyz"][mask_all], g.ndata["particle_number"][mask_all].long().view(-1), dim=0)
   
    pos_track = g.ndata["pos_hits_xyz"][mask_hit_type_t2]
    particle_track = g.ndata["particle_number"][mask_hit_type_t2]
    if  torch.sum(g.ndata["particle_number"] == 0)==0:
        #then index 1 is at 0 
        mean_pos_cluster = mean_pos_cluster[1:,:]
        particle_track = particle_track-1
    # print(mean_pos_cluster.shape, torch.unique(g.ndata["particle_number"]).shape)
    # print("mean_pos_cluster", mean_pos_cluster.shape)
    # print("particle_track", particle_track)
    # print("pos_track", pos_track.shape)
    if mean_pos_cluster.shape[0] == torch.unique(g.ndata["particle_number"]).shape:
        distance_track_cluster = torch.norm(mean_pos_cluster[particle_track.long()]-pos_track,dim=1)/1000
        # print("distance_track_cluster", distance_track_cluster)
        bad_tracks = distance_track_cluster>0.21
        index_bad_tracks = mask_hit_type_t2.nonzero().view(-1)[bad_tracks]
        g.ndata["particle_number"][index_bad_tracks]= 0 
    return g