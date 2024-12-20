from gatr import GATr, SelfAttentionConfig, MLPConfig
from gatr.interface import (
    embed_point,
    extract_scalar,
    extract_point,
    embed_scalar,
    embed_translation,
    extract_translation
)
import torch
import torch.nn as nn
from pydantic.v1.main import Model
import dgl
from xformers.ops.fmha import BlockDiagonalMask


class GATrModel(torch.nn.Module):
    def __init__(self, n_scalars, hidden_mv_channels, hidden_s_channels, blocks, embed_as_vectors):
        super().__init__()
        self.n_scalars = n_scalars
        self.hidden_mv_channels = hidden_mv_channels
        self.hidden_s_channels = hidden_s_channels
        self.blocks = blocks
        self.embed_as_vectors = embed_as_vectors
        self.gatr = GATr(
            in_mv_channels=1,
            out_mv_channels=1,
            hidden_mv_channels=hidden_mv_channels,
            in_s_channels=n_scalars,
            hidden_s_channels=hidden_s_channels,
            num_blocks=blocks,
            attention=SelfAttentionConfig(),  # Use default parameters for attention
            mlp=MLPConfig(),  # Use default parameters for MLP
        )
        self.batch_norm = nn.BatchNorm1d(self.input_dim, momentum=0.1)
        #self.clustering = nn.Linear(3, self.output_dim - 1, bias=False)
        self.beta = nn.Linear(n_scalars + 1, 1)

    def forward(self, data):
        # data: instance of EventBatch
        inputs_v = data.input_vectors
        inputs_scalar = data.input_scalars
        assert inputs_scalar.shape[1] == self.n_scalars
        if self.embed_as_vectors:
            velocities = embed_translation(inputs_v)
            embedded_inputs = (
                velocities
            )
        else:
            inputs = self.batch_norm(inputs_v)
            embedded_inputs = embed_point(inputs) #+ embed_scalar(inputs_scalar)
        embedded_inputs = embedded_inputs.unsqueeze(-2)  # (batch_size*num_points, 1, 16)
        mask = self.build_attention_mask(data.batch_idx)
        embedded_outputs, output_scalars = self.gatr(
            embedded_inputs, scalars=inputs_scalar, attention_mask=mask
        )
        #if self.embed_as_vectors:
        #    x_clusters = extract_translation(embedded_outputs)
        #else:
        #    x_clusters = extract_point(embedded_outputs)
        x_clusters = extract_point(embedded_outputs)
        original_scalar = extract_scalar(embedded_outputs)
        beta = self.beta(torch.cat([original_scalar, output_scalars], dim=1))
        x = torch.cat((x_clusters, beta.view(-1, 1)), dim=1)
        return x

    def build_attention_mask(self, batch_numbers):
        return BlockDiagonalMask.from_seqlens(
            torch.bincount(batch_numbers.long()).tolist()
        )

def get_model(args):
    return GATrModel(
        n_scalars=3,
        hidden_mv_channels=16,
        hidden_s_channels=64,
        blocks=10,
        embed_as_vectors=False,
    )
