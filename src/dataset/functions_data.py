import numpy as np
import torch
import dgl
#from torch_scatter import scatter_add, scatter_sum


def get_ratios(e_hits, part_idx, y):
    """Obtain the percentage of energy of the particle present in the hits

    Args:
        e_hits (_type_): _description_
        part_idx (_type_): _description_
        y (_type_): _description_

    Returns:
        _type_: _description_
    """
    energy_from_showers = scatter_sum(e_hits, part_idx.long(), dim=0)
    # y_energy = y[:, 3]
    y_energy = y.E
    energy_from_showers = energy_from_showers[1:]
    assert len(energy_from_showers) > 0
    return (energy_from_showers.flatten() / y_energy).tolist()


def get_number_hits(e_hits, part_idx):
    number_of_hits = scatter_sum(torch.ones_like(e_hits), part_idx.long(), dim=0)
    return (number_of_hits[1:].flatten()).tolist()

def get_e_reco(e_hits, part_idx):
    number_of_hits = scatter_sum(e_hits, part_idx.long(), dim=0)
    return number_of_hits[1:].flatten()

def get_number_of_daughters(hit_type_feature, hit_particle_link, daughters):
    a = hit_particle_link
    b = daughters
    a_u = torch.unique(a)
    number_of_p = torch.zeros_like(a_u)
    for p, i in enumerate(a_u):
        mask2 = a == i
        number_of_p[p] = torch.sum(torch.unique(b[mask2]) != -1)
    return number_of_p


def find_mask_no_energy(
    hit_particle_link,
    hit_type_a,
    hit_energies,
    y,
    daughters,
    predict=False,
    is_Ks=False,
):
    """This function remove particles with tracks only and remove particles with low fractions
    # Remove 2212 going to multiple particles without tracks for now
    # remove particles below energy cut
    # remove particles that decayed in the tracker
    # remove particles with two tracks (due to bad tracking)
    # remove particles with daughters for the moment

    Args:
        hit_particle_link (_type_): _description_
        hit_type_a (_type_): _description_
        hit_energies (_type_): _description_
        y (_type_): _description_

    Returns:
        _type_: _description_
    """

    number_of_daughters = get_number_of_daughters(
        hit_type_a, hit_particle_link, daughters
    )
    list_p = np.unique(hit_particle_link)
    list_remove = []
    part_frac = torch.tensor(get_ratios(hit_energies, hit_particle_link, y))
    number_of_hits = get_number_hits(hit_energies, hit_particle_link)
    if predict:
        energy_cut = 0.1
        filt1 = (torch.where(part_frac >= energy_cut)[0] + 1).long().tolist()
    else:
        energy_cut = 0.01
        filt1 = (torch.where(part_frac >= energy_cut)[0] + 1).long().tolist()
    number_of_tracks = scatter_add(1 * (hit_type_a == 1), hit_particle_link.long())[1:]
    if is_Ks == False:
        for index, p in enumerate(list_p):
            mask = hit_particle_link == p
            hit_types = np.unique(hit_type_a[mask])

            if predict:
                if (
                    np.array_equal(hit_types, [0, 1])
                    or int(p) not in filt1
                    or (number_of_hits[index] < 2)
                    or (y.decayed_in_tracker[index] == 1)
                    or number_of_tracks[index] == 2
                    or number_of_daughters[index] > 1
                ):
                    list_remove.append(p)
            else:
                if (
                    np.array_equal(hit_types, [0, 1])
                    or int(p) not in filt1
                    or (number_of_hits[index] < 2)
                    or number_of_tracks[index] == 2
                    or number_of_daughters[index] > 1
                ):
                    list_remove.append(p)
    if len(list_remove) > 0:
        mask = torch.tensor(np.full((len(hit_particle_link)), False, dtype=bool))
        for p in list_remove:
            mask1 = hit_particle_link == p
            mask = mask1 + mask

    else:
        mask = np.full((len(hit_particle_link)), False, dtype=bool)

    if len(list_remove) > 0:
        mask_particles = np.full((len(list_p)), False, dtype=bool)
        for p in list_remove:
            mask_particles1 = list_p == p
            mask_particles = mask_particles1 + mask_particles

    else:
        mask_particles = np.full((len(list_p)), False, dtype=bool)
    return mask, mask_particles


class CachedIndexList:
    def __init__(self, lst):
        self.lst = lst
        self.cache = {}

    def index(self, value):
        if value in self.cache:
            return self.cache[value]
        else:
            idx = self.lst.index(value)
            self.cache[value] = idx
            return idx


def find_cluster_id(hit_particle_link):
    unique_list_particles = list(np.unique(hit_particle_link))

    if np.sum(np.array(unique_list_particles) == -1) > 0:
        non_noise_idx = torch.where(hit_particle_link != -1)[0]  #
        noise_idx = torch.where(hit_particle_link == -1)[0]  #
        unique_list_particles1 = torch.unique(hit_particle_link)[1:]
        cluster_id_ = torch.searchsorted(
            unique_list_particles1, hit_particle_link[non_noise_idx], right=False
        )
        cluster_id_small = 1.0 * cluster_id_ + 1
        cluster_id = hit_particle_link.clone()
        cluster_id[non_noise_idx] = cluster_id_small
        cluster_id[noise_idx] = 0
    else:
        c_unique_list_particles = CachedIndexList(unique_list_particles)
        cluster_id = map(
            lambda x: c_unique_list_particles.index(x), hit_particle_link.tolist()
        )
        cluster_id = torch.Tensor(list(cluster_id)) + 1
    return cluster_id, unique_list_particles


def scatter_count(input: torch.Tensor):
    return scatter_add(torch.ones_like(input, dtype=torch.long), input.long())


def get_particle_features(unique_list_particles, output, prediction, connection_list):
    unique_list_particles = torch.Tensor(unique_list_particles).to(torch.int64)
    if prediction:
        number_particle_features = 12 - 2
    else:
        number_particle_features = 9 - 2
    if output["pf_features"].shape[0] == 18:
        number_particle_features += 8  # add vertex information
    features_particles = torch.permute(
        torch.tensor(
            output["pf_features"][
                2:number_particle_features, list(unique_list_particles)
            ]
        ),
        (1, 0),
    )  #
    # particle_coord are just features 10, 11, 12
    if features_particles.shape[1] == 16: # Using config with part_pxyz and part_vertex_xyz
        #print("Using config with part_pxyz and part_vertex_xyz")
        particle_coord = features_particles[:, 10:13]
        vertex_coord = features_particles[:, 13:16]
        # normalize particle coords
        particle_coord = particle_coord# / np.linalg.norm(particle_coord, axis=1).reshape(-1, 1)  # DO NOT NORMALIZE
        #particle_coord, spherical_to_cartesian(
        #    features_particles[:, 1],
        #    features_particles[:, 0],  # theta and phi are mixed!!!
        #    features_particles[:, 2],
        #    normalized=True,
        #)
    else:
        particle_coord = spherical_to_cartesian(
            features_particles[:, 1],
            features_particles[:, 0],  # theta and phi are mixed!!!
            features_particles[:, 2],
            normalized=True,
        )
        vertex_coord = torch.zeros_like(particle_coord)
    y_mass = features_particles[:, 3].view(-1).unsqueeze(1)
    y_mom = features_particles[:, 2].view(-1).unsqueeze(1)
    y_energy = torch.sqrt(y_mass**2 + y_mom**2)
    y_pid = features_particles[:, 4].view(-1).unsqueeze(1)
    if prediction:
        y_data_graph = Particles_GT(
            particle_coord,
            y_energy,
            y_mom,
            y_mass,
            y_pid,
            features_particles[:, 5].view(-1).unsqueeze(1),
            features_particles[:, 6].view(-1).unsqueeze(1),
            unique_list_particles=unique_list_particles,
            vertex=vertex_coord,
        )
    else:
        y_data_graph = Particles_GT(
            particle_coord,
            y_energy,
            y_mom,
            y_mass,
            y_pid,
            unique_list_particles=unique_list_particles,
            vertex=vertex_coord,
        )
    return y_data_graph


def modify_index_link_for_gamma_e(
    hit_type_feature, hit_particle_link, daughters, output, number_part, is_Ks=False
):
    """Split all particles that have daughters, mostly for brems and conversions but also for protons and neutrons

    Returns:
        hit_particle_link: new link
        hit_link_modified: bool for modified hits
    """
    hit_link_modified = torch.zeros_like(hit_particle_link).to(hit_particle_link.device)
    mask = hit_type_feature > 1
    a = hit_particle_link[mask]
    b = daughters[mask]
    a_u = torch.unique(a)
    number_of_p = torch.zeros_like(a_u)
    connections_list = []
    for p, i in enumerate(a_u):
        mask2 = a == i
        list_of_daugthers = torch.unique(b[mask2])
        number_of_p[p] = len(list_of_daugthers)
        if (number_of_p[p] > 1) and (torch.sum(list_of_daugthers == i) > 0):
            connections_list.append([i, torch.unique(b[mask2])])
    pid_particles = torch.tensor(output["pf_features"][6, 0:number_part])
    electron_photon_mask = (torch.abs(pid_particles[a_u.long()]) == 11) + (
        pid_particles[a_u.long()] == 22
    )
    electron_photon_mask = (
        electron_photon_mask * number_of_p > 1
    )  # electron_photon_mask *
    if is_Ks:
        index_change = a_u  # [electron_photon_mask]
    else:
        index_change = a_u[electron_photon_mask]
    for i in index_change:
        mask_n = mask * (hit_particle_link == i)
        hit_particle_link[mask_n] = daughters[mask_n]
        hit_link_modified[mask_n] = 1
    return hit_particle_link, hit_link_modified, connections_list


def get_hit_features(
    output, number_hits, prediction, number_part, hit_chis, pos_pxpy, is_Ks=False
):

    hit_particle_link = torch.tensor(output["pf_vectoronly"][0, 0:number_hits])
    if prediction:
        indx_daugthers = 3
    else:
        indx_daugthers = 1
    daughters = torch.tensor(output["pf_vectoronly"][indx_daugthers, 0:number_hits])
    if prediction:
        pandora_cluster = torch.tensor(output["pf_vectoronly"][1, 0:number_hits])
        pandora_pfo_link = torch.tensor(output["pf_vectoronly"][2, 0:number_hits])
        if is_Ks:
            pandora_mom = torch.permute(
                torch.tensor(output["pf_points_pfo"][0:3, 0:number_hits]), (1, 0)
            )
            pandora_ref_point = torch.permute(
                torch.tensor(output["pf_points_pfo"][3:6, 0:number_hits]), (1, 0)
            )
            if output["pf_points_pfo"].shape[0] > 6:
                pandora_pid = torch.tensor(output["pf_points_pfo"][6, 0:number_hits])
            else:
                # zeros
                # print("Zeros for pandora pid!")
                pandora_pid=torch.zeros(number_hits)

        else:
            pandora_mom = None
            pandora_ref_point = None
            pandora_pid = None
        if is_Ks:
            pandora_cluster_energy = torch.tensor(
                output["pf_features"][9, 0:number_hits]
            )
            pfo_energy = torch.tensor(output["pf_features"][10, 0:number_hits])
            chi_squared_tracks = torch.tensor(output["pf_features"][11, 0:number_hits])
        elif hit_chis:
            pandora_cluster_energy = torch.tensor(
                output["pf_features"][-3, 0:number_hits]
            )
            pfo_energy = torch.tensor(output["pf_features"][-2, 0:number_hits])
            chi_squared_tracks = torch.tensor(output["pf_features"][-1, 0:number_hits])
        else:
            pandora_cluster_energy = torch.tensor(
                output["pf_features"][-2, 0:number_hits]
            )
            pfo_energy = torch.tensor(output["pf_features"][-1, 0:number_hits])
            chi_squared_tracks = None

    else:
        pandora_cluster = None
        pandora_pfo_link = None
        pandora_cluster_energy = None
        pfo_energy = None
        chi_squared_tracks = None
        pandora_mom = None
        pandora_ref_point = None
        pandora_pid = None
    # hit type
    hit_type_feature = torch.permute(
        torch.tensor(output["pf_vectors"][:, 0:number_hits]), (1, 0)
    )[:, 0].to(torch.int64)

    (
        hit_particle_link,
        hit_link_modified,
        connection_list,
    ) = modify_index_link_for_gamma_e(
        hit_type_feature, hit_particle_link, daughters, output, number_part, is_Ks
    )

    cluster_id, unique_list_particles = find_cluster_id(hit_particle_link)

    # position, e, p
    pos_xyz_hits = torch.permute(
        torch.tensor(output["pf_points"][0:3, 0:number_hits]), (1, 0)
    )
    pf_features_hits = torch.permute(
        torch.tensor(output["pf_features"][0:2, 0:number_hits]), (1, 0)
    )  # removed theta, phi
    p_hits = pf_features_hits[:, 0].unsqueeze(1)
    p_hits[p_hits == -1] = 0  # correct p  of Hcal hits to be 0
    e_hits = pf_features_hits[:, 1].unsqueeze(1)
    e_hits[e_hits == -1] = 0  # correct the energy of the tracks to be 0
    if pos_pxpy:
        pos_pxpypz = torch.permute(
            torch.tensor(output["pf_points"][3:, 0:number_hits]), (1, 0)
        )
    else:
        pos_pxpypz = pos_xyz_hits
    # pos_pxpypz = pos_theta_phi

    return (
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
    )


def standardize_coordinates(coord_cart_hits):
    if len(coord_cart_hits) == 0:
        return coord_cart_hits, None
    std_scaler = StandardScaler()
    coord_cart_hits = std_scaler.fit_transform(coord_cart_hits)
    return torch.tensor(coord_cart_hits).float(), std_scaler


def create_dif_interactions(i, j, pos, number_p):
    x_interactions = pos
    x_interactions = torch.reshape(x_interactions, [number_p, 1, 2])
    x_interactions = x_interactions.repeat(1, number_p, 1)
    xi = x_interactions[i, j, :]
    xj = x_interactions[j, i, :]
    x_interactions_m = xi - xj
    return x_interactions_m


def spherical_to_cartesian(phi, theta, r, normalized=False):
    if normalized:
        r = torch.ones_like(phi)
    x = r * torch.sin(theta) * torch.cos(phi)
    y = r * torch.sin(theta) * torch.sin(phi)
    z = r * torch.cos(theta)
    return torch.cat((x.unsqueeze(1), y.unsqueeze(1), z.unsqueeze(1)), dim=1)


def calculate_distance_to_boundary(g):
    r = 2150
    r_in_endcap = 2307
    mask_endcap = (torch.abs(g.ndata["pos_hits_xyz"][:, 2]) - r_in_endcap) > 0
    mask_barrer = ~mask_endcap
    weight = torch.ones_like(g.ndata["pos_hits_xyz"][:, 0])
    C = g.ndata["pos_hits_xyz"]
    A = torch.Tensor([0, 0, 1]).to(C.device)
    P = (
        r
        * 1
        / (torch.norm(torch.cross(A.view(1, -1), C, dim=-1), dim=1)).unsqueeze(1)
        * C
    )
    P1 = torch.abs(r_in_endcap / g.ndata["pos_hits_xyz"][:, 2].unsqueeze(1)) * C
    weight[mask_barrer] = torch.norm(P - C, dim=1)[mask_barrer]
    weight[mask_endcap] = torch.norm(P1[mask_endcap] - C[mask_endcap], dim=1)
    g.ndata["radial_distance"] = weight
    weight_ = torch.exp(-(weight / 1000))
    g.ndata["radial_distance_exp"] = weight_
    return g

class EventCollection:
    def mask(self, mask):
        for k in self.__dict__:
            if getattr(self, k) is not None:
                if type(getattr(self, k)) == list:
                    if getattr(self, k)[0] is not None:
                        setattr(self, k, getattr(self, k)[mask])
                elif not type(getattr(self, k)) == dict:
                    setattr(self, k, getattr(self, k)[mask])
                else:
                    raise NotImplementedError("Need to implement correct indexing")
                    # TODO: for the mapping pfcands_idx to jet_idx

    def copy(self):
        obj = type(self).__new__(self.__class__)
        obj.__dict__.update(self.__dict__)
        return obj

    def serialize(self):
        # get all the self.init_attrs and concat them together. Also return batch_number
        data = torch.stack([getattr(self, attr) for attr in type(self).init_attrs]).T
        assert data.shape[0] == self.batch_number.max().item()
        return data, self.batch_number

    @staticmethod
    def deserialize(data_matrix, batch_number, cls):
        data = {}
        for i, key in enumerate(cls.init_attrs):
            data[key] = data_matrix[:, i]
        return cls(**data, batch_number=batch_number)


def concat_event_collection(list_event_collection):
    c = list_event_collection[0]
    list_of_attrs = c.init_attrs
    #for k in c.__dict__:
    #    if getattr(c, k) is not None:
    #        if isinstance(getattr(c, k), torch.Tensor):
    #            list_of_attrs.append(k)
    result = {}
    for attr in list_of_attrs:
        result[attr] = torch.cat([getattr(c, attr) for c in list_event_collection], dim=0)
    batch_number = add_batch_number(list_event_collection, attr=list_of_attrs[0])
    return type(c)(**result, batch_number=batch_number)

def concat_events(list_events):
    attrs = list_events[0].init_attrs
    result = {}
    for attr in attrs:
        result[attr] = concat_event_collection([getattr(e, attr) for e in list_events])
        # assert result[attr].batch_number.max() == len(list_events)# sometimes the event is empty (e.g. no found jets)
    return Event(**result, n_events=len(list_events))

def get_batch(event, batch_config):
    # Returns the EventBatch class, with correct scalars etc.
    batch_idx_pfcands = torch.zeros(len(event.pfcands)).long()
    batch_idx_special_pfcands = torch.zeros(len(event.special_pfcands)).long()
    for i in range(len(event.pfcands.number_batch)):
        batch_idx_pfcands[event.pfcands.number_batch[i]:event.pfcands.number_batch[i+1]] = i
    for i in range(len(event.special_pfcands.number_batch)):
        batch_idx_special_pfcands[event.special_pfcands.number_batch[i]:event.special_pfcands.number_batch[i+1]] = i
    batch_idx = torch.cat([batch_idx_pfcands, batch_idx_special_pfcands])
    batch_idx = batch_idx.to(event.pfcands.pt.device)
    if batch_config.get("use_p_xyz", True):
        batch_vectors = torch.cat([event.pfcands.pxyz, event.special_pfcands.pxyz], dim=0)
    else:
        raise NotImplementedError
    pids = batch_config.get("pids", [11, 13, 22, 130, 211, 0, 1, 2, 3]) # 0, 1, 2, 3 are the special PFcands
    # onehot encode pids of event.pfcands.pid
    pids_onehot = torch.zeros(len(event.pfcands), len(pids))
    for i, pid in enumerate(pids):
        pids_onehot[:, i] = (event.pfcands.pid == pid).float()
    assert (pids_onehot.sum(dim=1) == 1).all()
    batch_scalars_pfcands = torch.cat([event.pfcands.charge, pids_onehot], dim=1)
    batch_scalars_special_pfcands = torch.cat([event.special_pfcands.charge, event.special_pfcands.pid.float().unsqueeze(1)], dim=1)
    batch_scalars = torch.cat([batch_scalars_pfcands, batch_scalars_special_pfcands], dim=0)
    assert batch_idx.max() == event.n_events - 1
    return EventBatch(
        input_vectors=batch_vectors,
        input_scalars=batch_scalars,
        batch_idx=batch_idx,
    )

def to_tensor(item):
    if isinstance(item, torch.Tensor):
        return item
    return torch.tensor(item)

class EventPFCands(EventCollection):
    init_attrs = ["pt", "eta", "phi", "mass", "charge", "pid", "pf_cand_jet_idx"]
    def __init__(
        self,
        pt,
        eta,
        phi,
        mass,
        charge,
        pid,
        jet_idx=None,
        pfcands_idx=None,
        batch_number=None,
        offline=False,
        pf_cand_jet_idx=None, # optional: provide either this or pfcands_idx & jet_idx
    ):
        #print("Jet idx:", jet_idx)
        #print("PFCands_idx:", pfcands_idx)
        self.pt = to_tensor(pt)
        self.eta = to_tensor(eta)
        self.theta = 2 * torch.atan(torch.exp(-self.eta))
        self.p = pt / torch.sin(self.theta)
        self.phi = to_tensor(phi)
        self.pxyz = torch.stack(
      (self.p * torch.cos(self.phi) * torch.sin(self.theta),
             self.p * torch.sin(self.phi) * torch.sin(self.theta),
             self.p * torch.cos(self.theta)),
            dim=1
        )
        assert (torch.abs(torch.norm(self.pxyz, dim=1) - self.p) < 1e-3).all()
        self.mass = to_tensor(mass)
        self.E = torch.sqrt(self.mass ** 2 + self.p ** 2)
        self.charge = to_tensor(charge)
        self.pid = to_tensor(pid)
        if pf_cand_jet_idx is not None:
            self.pf_cand_jet_idx = to_tensor(pf_cand_jet_idx)
        else:
            self.pf_cand_jet_idx = torch.ones(len(self.pt)).int() * -1
            for i, pfcand_idx in enumerate(pfcands_idx):
                if int(pfcand_idx) >= len(self.pt):
                    print("Out of bounds")
                    if not offline:
                        raise Exception
                else:
                    self.pf_cand_jet_idx[int(pfcand_idx)] = int(jet_idx[i])
        if batch_number is not None:
            self.batch_number = batch_number
    def __len__(self):
        return len(self.pt)

class EventMetadataAndMET(EventCollection):
    init_attrs = ["pt", "phi", "scouting_trig", "offline_trig", "veto_trig"]
    # Extra info belonging to the event: MET, trigger info etc.
    def __init__(self, pt, phi, scouting_trig, offline_trig, veto_trig, batch_number=None):
        self.pt = to_tensor(pt)
        self.phi = to_tensor(phi)
        self.scouting_trig = to_tensor(scouting_trig)
        self.offline_trig = to_tensor(offline_trig)
        self.veto_trig = to_tensor(veto_trig)
        if batch_number is not None:
            self.batch_number = to_tensor(batch_number)
    def __len__(self):
        return len(self.pt)

class EventJets(EventCollection):
    init_attrs = ["pt", "eta", "phi", "mass"]
    def __init__(
        self,
        pt,
        eta,
        phi,
        mass,
        area=None,
        batch_number=None
    ):
        self.pt = to_tensor(pt)
        self.eta = to_tensor(eta)
        self.theta = 2 * torch.atan(torch.exp(-self.eta))
        self.p = pt / torch.sin(self.theta)
        self.phi = to_tensor(phi)
        self.pxyz = torch.stack(
      (self.p * torch.cos(self.phi) * torch.sin(self.theta),
             self.p * torch.sin(self.phi) * torch.sin(self.theta),
             self.p * torch.cos(self.theta)),
            dim=1
        )
        assert (torch.abs(torch.norm(self.pxyz, dim=1) - self.p) < 1e-3).all()
        self.mass = to_tensor(mass)
        self.area = area
        if self.area is not None:
            self.area = to_tensor(self.area)
        if batch_number is not None:
            self.batch_number = to_tensor(batch_number)

    def __len__(self):
        return len(self.pt)


class Particles_GT:
    def __init__(
        self,
        coordinates,
        energy,
        momentum,
        mass,
        pid,
        decayed_in_calo=None,
        decayed_in_tracker=None,
        batch_number=None,
        unique_list_particles=None,
        energy_corrected=None,
        vertex=None,
    ):
        self.coord = coordinates
        self.E = energy
        self.E_corrected = energy
        if energy_corrected is not None:
            self.E_corrected = energy_corrected
        assert len(coordinates) == len(energy)
        self.m = momentum
        self.mass = mass
        self.pid = pid
        self.vertex = vertex
        if unique_list_particles is not None:
            self.unique_list_particles = unique_list_particles
        if decayed_in_calo is not None:
            self.decayed_in_calo = decayed_in_calo
        if decayed_in_tracker is not None:
            self.decayed_in_tracker = decayed_in_tracker
        if batch_number is not None:
            self.batch_number = batch_number

    def __len__(self):
        return len(self.E)

    def mask(self, mask):
        for k in self.__dict__:
            if getattr(self, k) is not None:
                if type(getattr(self, k)) == list:
                    if getattr(self, k)[0] is not None:
                        setattr(self, k, getattr(self, k)[mask])
                else:
                    setattr(self, k, getattr(self, k)[mask])

    def copy(self):
        obj = type(self).__new__(self.__class__)
        obj.__dict__.update(self.__dict__)
        return obj

    def calculate_corrected_E(self, g, connections_list):
        for element in connections_list:
            # checked there is track
            parent_particle = element[0]
            mask_i = g.ndata["particle_number_nomap"] == parent_particle
            track_number = torch.sum(g.ndata["hit_type"][mask_i] == 1)
            if track_number > 0:
                # find index in list
                index_parent = torch.argmax(
                    1 * (self.unique_list_particles == parent_particle)
                )
                energy_daugthers = 0
                for daugther in element[1]:
                    if daugther != parent_particle:
                        if torch.sum(self.unique_list_particles == daugther) > 0:
                            index_daugthers = torch.argmax(
                                1 * (self.unique_list_particles == daugther)
                            )
                            energy_daugthers = (
                                self.E[index_daugthers] + energy_daugthers
                            )
                self.E_corrected[index_parent] = (
                    self.E_corrected[index_parent] - energy_daugthers
                )
                self.coord[index_parent] *= (1 - energy_daugthers / torch.norm(self.coord[index_parent]))

def concatenate_Particles_GT(list_of_Particles_GT):
    list_coord = [p[1].coord for p in list_of_Particles_GT]
    list_vertex = [p[1].vertex for p in list_of_Particles_GT]
    list_coord = torch.cat(list_coord, dim=0)
    list_E = [p[1].E for p in list_of_Particles_GT]
    list_E = torch.cat(list_E, dim=0)
    list_E_corr = [p[1].E_corrected for p in list_of_Particles_GT]
    list_E_corr = torch.cat(list_E_corr, dim=0)
    list_m = [p[1].m for p in list_of_Particles_GT]
    list_m = torch.cat(list_m, dim=0)
    list_mass = [p[1].mass for p in list_of_Particles_GT]
    list_mass = torch.cat(list_mass, dim=0)
    list_pid = [p[1].pid for p in list_of_Particles_GT]
    list_pid = torch.cat(list_pid, dim=0)
    if list_vertex[0] is not None:
        list_vertex = torch.cat(list_vertex, dim=0)
    if hasattr(list_of_Particles_GT[0], "decayed_in_calo"):
        list_dec_calo = [p[1].decayed_in_calo for p in list_of_Particles_GT]
        list_dec_track = [p[1].decayed_in_tracker for p in list_of_Particles_GT]
        list_dec_calo = torch.cat(list_dec_calo, dim=0)
        list_dec_track = torch.cat(list_dec_track, dim=0)
    else:
        list_dec_calo = None
        list_dec_track = None
    batch_number = add_batch_number(list_of_Particles_GT)
    return Particles_GT(
        list_coord,
        list_E,
        list_m,
        list_mass,
        list_pid,
        list_dec_calo,
        list_dec_track,
        batch_number,
        energy_corrected=list_E_corr,
        vertex=list_vertex,
    )

def add_batch_number(list_event_collections, attr):
    list_y = []
    idx = 0
    list_y.append(idx)
    for i, el in enumerate(list_event_collections):
        num_in_batch = el.__dict__[attr].shape[0]
        list_y.append(idx + num_in_batch)
        idx += num_in_batch
    list_y = torch.tensor(list_y)
    return list_y

def create_noise_label(hit_energies, hit_particle_link, y, cluster_id):
    unique_p_numbers = torch.unique(cluster_id)
    number_of_hits = get_number_hits(hit_energies, cluster_id)
    e_reco = get_e_reco(hit_energies, cluster_id)
    mask_hits = to_tensor(number_of_hits) < 6
    mask_p = e_reco<0.10
    mask_all = mask_hits.view(-1) + mask_p.view(-1)
    list_remove = unique_p_numbers[mask_all.view(-1)]

    if len(list_remove) > 0:
        mask = to_tensor(np.full((len(cluster_id)), False, dtype=bool))
        for p in list_remove:
            mask1 = cluster_id == p
            mask = mask1 + mask
    else:
        mask = to_tensor(np.full((len(cluster_id)), False, dtype=bool))
    list_p = unique_p_numbers
    if len(list_remove) > 0:
        mask_particles = np.full((len(list_p)), False, dtype=bool)
        for p in list_remove:
            mask_particles1 = list_p == p
            mask_particles = mask_particles1 + mask_particles
    else:
        mask_particles = to_tensor(np.full((len(list_p)), False, dtype=bool))
    return mask.to(bool), ~mask_particles.to(bool)

class EventBatch:
    def __init__(self, input_vectors, input_scalars, batch_idx):
        self.input_vectors = input_vectors
        self.input_scalars = input_scalars
        self.batch_idx = batch_idx
    def to(self, device):
        self.input_vectors = self.input_vectors.to(device)
        self.input_scalars = self.input_scalars.to(device)
        self.batch_idx = self.batch_idx.to(device)

class Event:
    evt_collections = {"jets": EventJets, "genjets": EventJets, "pfcands": EventPFCands,
                       "offline_pfcands": EventPFCands, "MET": EventMetadataAndMET, "fatjets": EventJets,
                       "special_pfcands": EventPFCands, "matrix_element_gen_particles": EventPFCands}
    def __init__(self, jets=None, genjets=None, pfcands=None, offline_pfcands=None, MET=None, fatjets=None,
                 special_pfcands=None, matrix_element_gen_particles=None, n_events=1): # Add more collections here
        self.jets = jets
        self.genjets = genjets
        self.pfcands = pfcands
        self.offline_pfcands = offline_pfcands
        self.MET = MET
        self.fatjets = fatjets
        self.special_pfcands = special_pfcands
        self.matrix_element_gen_particles = matrix_element_gen_particles
        self.init_attrs = []
        self.n_events = n_events
        if jets is not None:
            self.init_attrs.append("jets")
        if genjets is not None:
            self.init_attrs.append("genjets")
        if pfcands is not None:
            self.init_attrs.append("pfcands")
        if offline_pfcands is not None:
            self.init_attrs.append("offline_pfcands")
        if MET is not None:
            self.init_attrs.append("MET")
        if fatjets is not None:
            self.init_attrs.append("fatjets")
        if special_pfcands is not None:
            self.init_attrs.append("special_pfcands")
        if matrix_element_gen_particles is not None:
            self.init_attrs.append("matrix_element_gen_particles")
    ''' @staticmethod
    def deserialize(result, result_metadata, event_idx=None):
        # 'result' arrays can be mmap-ed.
        # If event_idx is not None and is set to a list, only the selected event_idx will be returned.
        n_events = result_metadata["n_events"]
        attrs = result.keys()
        if event_idx is None:
            event_idx = to_tensor(list(range(n_events)))
        else:
            event_idx = to_tensor(event_idx)
            assert (event_idx < n_events).all()
        return Event(**{key: result[key][torch.isin(result_metadata[key + "_batch_idx"], event_idx)] for key in attrs}, n_events=n_events)
    '''
    def __len__(self):
        return self.n_events
    def serialize(self):
        result = {}
        result_metadata = {"n_events": self.n_events, "attrs": self.init_attrs}
        for key in self.init_attrs:
            s = getattr(self, key).serialize()
            result[key] = s[0]
            result_metadata[key + "_batch_idx"] = s[1]
        return result, result_metadata
