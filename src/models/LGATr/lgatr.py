from lgatr import GATr, SelfAttentionConfig, MLPConfig
from lgatr.interface import embed_vector, extract_scalar, embed_spurions, extract_vector
import torch
import torch.nn as nn
from xformers.ops.fmha import BlockDiagonalMask
from torch_scatter import scatter_sum, scatter_max, scatter_mean


class LGATrModel(torch.nn.Module):
    def __init__(self, n_scalars, hidden_mv_channels, hidden_s_channels, blocks, embed_as_vectors, n_scalars_out, return_scalar_coords, obj_score=False, global_featuers_copy=False):
        super().__init__()
        self.return_scalar_coords = return_scalar_coords
        self.n_scalars = n_scalars
        self.hidden_mv_channels = hidden_mv_channels
        self.hidden_s_channels = hidden_s_channels
        self.blocks = blocks
        self.embed_as_vectors = embed_as_vectors
        self.input_dim = 3
        self.n_scalars_out = n_scalars_out
        self.obj_score = obj_score
        self.global_features_copy = global_featuers_copy
        self.gatr = GATr(
            in_mv_channels=3,
            out_mv_channels=1,
            hidden_mv_channels=hidden_mv_channels,
            in_s_channels=n_scalars,
            out_s_channels=n_scalars_out,
            hidden_s_channels=hidden_s_channels,
            num_blocks=blocks,
            attention=SelfAttentionConfig(),  # Use default parameters for attention
            mlp=MLPConfig(),  # Use default parameters for MLP
        )
        if self.global_features_copy:
            self.gatr_global_features = GATr(
                in_mv_channels=3,
                out_mv_channels=1,
                hidden_mv_channels=hidden_mv_channels,
                in_s_channels=n_scalars,
                out_s_channels=n_scalars_out,
                hidden_s_channels=hidden_s_channels,
                num_blocks=blocks,
                attention=SelfAttentionConfig(),  # Use default parameters for attention
                mlp=MLPConfig(),  # Use default parameters for MLP
            )
        #self.batch_norm = nn.BatchNorm1d(self.input_dim, momentum=0.1)
        #self.clustering = nn.Linear(3, self.output_dim - 1, bias=False)
        if n_scalars_out > 0:
            if obj_score:
                factor = 1
                if self.global_features_copy: factor = 2
                self.beta = nn.Sequential(
                    nn.Linear((n_scalars_out + 1) * factor, 10),
                    nn.LeakyReLU(),
                    nn.Linear(10, 1),
                    #nn.Sigmoid()
                )
            else:
                self.beta = nn.Linear(n_scalars_out + 1, 1)
        else:
            self.beta = None

    def forward(self, data, data_events=None, data_events_clusters=None):
        # data: instance of EventBatch
        if self.global_features_copy:
            assert data_events is not None and data_events_clusters is not None
            assert self.obj_score
            inputs_v = data_events.input_vectors
            inputs_scalar = data_events.input_scalars
            assert inputs_scalar.shape[1] == self.n_scalars, "Expected %d, got %d" % (
            self.n_scalars, inputs_scalar.shape[1])
            mask_global = self.build_attention_mask(data_events.batch_idx)
            embedded_inputs_events = embed_vector(inputs_v.unsqueeze(0))
            multivectors = embedded_inputs_events.unsqueeze(-2)
            spurions = embed_spurions(beam_reference="xyplane", add_time_reference=True,
                                      device=multivectors.device, dtype=multivectors.dtype)

            num_points, x = inputs_v.shape
            assert x == 4
            spurions = spurions[None, None, ...].repeat(1, num_points, 1, 1)  # (batchsize, num_points, 2, 16)
            multivectors = torch.cat((multivectors, spurions), dim=-2)
            embedded_outputs, output_scalars = self.gatr_global_features(
                multivectors, scalars=inputs_scalar, attention_mask=mask_global
            )
            original_scalar = extract_scalar(embedded_outputs)
            scalar_embeddings_nodes = torch.cat([original_scalar[0, :, 0, :], output_scalars[0, :, :]], dim=1)
            scalar_embeddings_global = scatter_mean(scalar_embeddings_nodes, torch.tensor(data_events_clusters).to(scalar_embeddings_nodes.device)+1, dim=0)[1:]

        inputs_v = data.input_vectors # four-momenta
        inputs_scalar = data.input_scalars
        assert inputs_scalar.shape[1] == self.n_scalars
        num_points, x = inputs_v.shape
        assert x == 4
        #velocities = embed_vector(inputs_v)
        inputs_v = inputs_v.unsqueeze(0)
        embedded_inputs = embed_vector(inputs_v)
        # if it contains nans, raise an error
        if torch.isnan(embedded_inputs).any():
            raise ValueError("NaNs in the input!")
        multivectors = embedded_inputs.unsqueeze(-2) # (batch_size*num_points, 1, 16)
        # for spurions, duplicate each unique batch_idx. e.g. [0,0,1,1,2,2] etc.
        #spurions_batch_idx = torch.repeat_interleave(data.batch_idx.unique(), 2)
        #batch_idx = torch.cat([data.batch_idx, spurions_batch_idx])
        spurions = embed_spurions(beam_reference="xyplane", add_time_reference=True,
                                  device=multivectors.device, dtype=multivectors.dtype)
        spurions = spurions[None, None, ...].repeat(1, num_points, 1, 1)  # (batchsize, num_points, 2, 16)
        multivectors = torch.cat((multivectors, spurions), dim=-2)  # (batchsize, num_points, 3, 16) - just embed the spurions as two extra multivector channels
        mask = self.build_attention_mask(data.batch_idx)
        embedded_outputs, output_scalars = self.gatr(
            multivectors, scalars=inputs_scalar, attention_mask=mask
        )

        #if self.embed_as_vectors:
        #    x_clusters = extract_translation(embedded_outputs)
        #else:
        #    x_clusters = extract_point(embedded_outputs)
        x_clusters = extract_vector(embedded_outputs)
        original_scalar = extract_scalar(embedded_outputs)
        if self.beta is not None:
            if self.obj_score:
                extract_from_virtual_nodes = False
                # assert that data has fake_nodes_idx from which we read the objectness score
                #assert "fake_nodes_idx" in data.__dict__
                # print batch number 3 and 4 inputs
                #for nbatch in [3, 4]:
                #    print("#### batch no. ", nbatch , "#######")
               #     print(" -> scalar inputs", inputs_scalar[data.batch_idx==nbatch].shape, inputs_scalar[data.batch_idx == nbatch])
               #     print(" -> vector inputs", data.input_vectors[data.batch_idx==nbatch].shape, data.input_vectors[data.batch_idx == nbatch])
               #     print("############")
                scalar_embeddings = torch.cat([original_scalar[0, :, 0, :], output_scalars[0, :, :]], dim=1)
                if extract_from_virtual_nodes:
                    values = torch.cat([original_scalar[0, data.fake_nodes_idx, 0, :], output_scalars[0, data.fake_nodes_idx, :]], dim=1)
                else:
                    values = scatter_mean(scalar_embeddings, data.batch_idx.to(scalar_embeddings.device).long(), dim=0)
                if self.global_features_copy:
                    values = torch.cat([values, scalar_embeddings_global], dim=1)
                beta = self.beta(values)
                #beta = self.beta(values)
                return beta
            vals = torch.cat([original_scalar[0, :, 0, :], output_scalars[0, :, :]], dim=1)
            beta = self.beta(vals)
            if self.return_scalar_coords:
                x = output_scalars[0, :, :3]
                #print(x.shape)
                #print(x[:5])
                x = torch.cat((x, torch.sigmoid(beta.view(-1, 1))), dim=1)
            else:
                x = torch.cat((x_clusters[0, :, 0, :], torch.sigmoid(beta.view(-1, 1))), dim=1)
        else:
            x = x_clusters[:, 0, :]
        if torch.isnan(x).any():
            raise ValueError("NaNs in the output!")
        #print(x[:5])
        return x

    def build_attention_mask(self, batch_numbers):
        return BlockDiagonalMask.from_seqlens(
            torch.bincount(batch_numbers.long()).tolist()
        )

def get_model(args, obj_score=False):
    n_scalars_out = 8
    if args.beta_type == "pt":
        n_scalars_out = 0
    elif args.beta_type == "pt+bc":
        n_scalars_out = 8
    n_scalars_in = 12
    if args.no_pid:
        n_scalars_in = 12 - 9
    if obj_score:
        return LGATrModel(
            n_scalars=n_scalars_in,
            hidden_mv_channels=8,
            hidden_s_channels=16,
            blocks=5,
            embed_as_vectors=False,
            n_scalars_out=n_scalars_out,
            return_scalar_coords=args.scalars_oc,
            obj_score=obj_score,
            global_featuers_copy=args.global_features_obj_score
        )

    return LGATrModel(
        n_scalars=n_scalars_in,
        hidden_mv_channels=args.hidden_mv_channels,
        hidden_s_channels=args.hidden_s_channels,
        blocks=args.num_blocks,
        embed_as_vectors=args.embed_as_vectors,
        n_scalars_out=n_scalars_out,
        return_scalar_coords=args.scalars_oc,
        obj_score=obj_score
    )

