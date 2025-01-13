from lgatr import GATr, SelfAttentionConfig, MLPConfig
from lgatr.interface import embed_vector, extract_scalar, embed_spurions, extract_vector
import torch
import torch.nn as nn
from xformers.ops.fmha import BlockDiagonalMask


class LGATrModel(torch.nn.Module):
    def __init__(self, n_scalars, hidden_mv_channels, hidden_s_channels, blocks, embed_as_vectors, n_scalars_out):
        super().__init__()
        self.n_scalars = n_scalars
        self.hidden_mv_channels = hidden_mv_channels
        self.hidden_s_channels = hidden_s_channels
        self.blocks = blocks
        self.embed_as_vectors = embed_as_vectors
        self.input_dim = 3
        self.n_scalars_out = n_scalars_out
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
        #self.batch_norm = nn.BatchNorm1d(self.input_dim, momentum=0.1)
        #self.clustering = nn.Linear(3, self.output_dim - 1, bias=False)
        if n_scalars_out > 0:
            self.beta = nn.Linear(n_scalars_out + 1, 1)
        else:
            self.beta = None

    def forward(self, data):
        # data: instance of EventBatch
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
        multivectors = torch.cat((multivectors, spurions), dim=-2)  # (batchsize, num_points, 3, 16)
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
            beta = self.beta(torch.cat([original_scalar[0, :, 0, :], output_scalars[0, :, :]], dim=1))
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

def get_model(args):
    n_scalars_out = 8
    if args.beta_type == "pt":
        n_scalars_out = 0
    elif args.beta_type == "pt+bc":
        n_scalars_out = 8
    return LGATrModel(
        n_scalars=12,
        hidden_mv_channels=16,
        hidden_s_channels=64,
        blocks=10,
        embed_as_vectors=args.embed_as_vectors,
        n_scalars_out=n_scalars_out
    )
