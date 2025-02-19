from src.models.transformer.tr_blocks import Transformer
import torch
import torch.nn as nn
from xformers.ops.fmha import BlockDiagonalMask
from torch_scatter import scatter_max, scatter_add, scatter_mean
import numpy as np


class TransformerModel(torch.nn.Module):
    def __init__(self, n_scalars, n_scalars_out, n_blocks, n_heads, internal_dim, obj_score, global_features_copy=False):
        super().__init__()
        self.n_scalars = n_scalars
        self.input_dim = n_scalars + 3
        if obj_score:
            self.input_dim += 1
        self.output_dim = 3
        self.obj_score = obj_score
        #internal_dim = 128
        #self.custom_decoder = nn.Linear(internal_dim, self.output_dim)
        #n_heads = 4
        #self.transformer = nn.TransformerEncoder(
        #    nn.TransformerEncoderLayer(
        #        d_model=n_heads*self.input_dim,
        #        nhead=n_heads,
        #        dim_feedforward=internal_dim,
        #        dropout=0.1,
        #        activation="gelu",
        #    ),
        #    num_layers=4,
        #)
        if n_scalars_out > 0:
            self.output_dim += 1 # betas regression
        if self.obj_score:
            self.output_dim = 10
        self.global_features_copy = global_features_copy
        self.transformer = Transformer(
            in_channels=self.input_dim,
            out_channels=self.output_dim,
            hidden_channels=internal_dim,
            num_heads=n_heads,
            num_blocks=n_blocks,
        )
        if self.global_features_copy:
            self.transformer_global_features = Transformer(
                in_channels=self.input_dim,
                out_channels=self.output_dim,
                hidden_channels=internal_dim,
                num_heads=n_heads,
                num_blocks=n_blocks,
            )
        self.batch_norm = nn.BatchNorm1d(self.input_dim, momentum=0.1)
        if self.obj_score:
            factor = 1
            if self.global_features_copy: factor = 2
            self.final_mlp = nn.Sequential(
                nn.Linear(self.output_dim*factor, 10),
                nn.LeakyReLU(),
                nn.Linear(10, 1),
            )
        #self.clustering = nn.Linear(3, self.output_dim - 1, bias=False)

    def forward(self, data, data_events=None, data_events_clusters=None):
        # data: instance of EventBatch
        # data_events & data_events_clusters: Only relevant if --global-features-obj-score is on: data_events contains
        # the "unmodified" batch where the batch indices are
        if self.global_features_copy:
            assert data_events is not None and data_events_clusters is not None
            assert self.obj_score
            inputs_v = data_events.input_vectors
            inputs_scalar = data_events.input_scalars
            assert inputs_scalar.shape[1] == self.n_scalars, "Expected %d, got %d" % (
            self.n_scalars, inputs_scalar.shape[1])
            inputs_transformer_events = torch.cat([inputs_scalar, inputs_v], dim=1)
            assert inputs_transformer_events.shape[1] == self.input_dim
            mask_global = self.build_attention_mask(data_events.batch_idx)
            x_global = inputs_transformer_events.unsqueeze(0)
            x_global = self.transformer_global_features(x_global, attention_mask=mask_global)[0]
            assert x_global.shape[1] == self.output_dim, "Expected %d, got %d" % (self.output_dim, x_global.shape[1])
            assert x_global.shape[0] == x_global.shape[0], "Expected %d, got %d" % (
            inputs_transformer_events.shape[0], x_global.shape[0])
            m_global = scatter_mean(x_global, torch.tensor(data_events_clusters).to(x_global.device)+1, dim=0)[1:]
        inputs_v = data.input_vectors
        inputs_scalar = data.input_scalars
        assert inputs_scalar.shape[1] == self.n_scalars, "Expected %d, got %d" % (self.n_scalars, inputs_scalar.shape[1])
        inputs_transformer = torch.cat([inputs_scalar, inputs_v], dim=1)
        print("input_dim", self.input_dim, inputs_transformer.shape)
        assert inputs_transformer.shape[1] == self.input_dim
        mask = self.build_attention_mask(data.batch_idx)
        x = inputs_transformer.unsqueeze(0)
        x = self.transformer(x, attention_mask=mask)[0]
        assert x.shape[1] == self.output_dim, "Expected %d, got %d" % (self.output_dim, x.shape[1])
        assert x.shape[0] == inputs_transformer.shape[0], "Expected %d, got %d" % (inputs_transformer.shape[0], x.shape[0])
        if not self.obj_score:
            x[:, -1] = torch.sigmoid(x[:, -1])
        else:
            extract_from_virtual_nodes = False
            if extract_from_virtual_nodes:
                x = self.final_mlp(x[data.fake_nodes_idx]) # x is the raw logits
            else:
                m = scatter_mean(x, torch.tensor(data.batch_idx).long().to(x.device), dim=0)
                assert not "fake_nodes_idx" in data.__dict__
                if self.global_features_copy:
                    m = torch.cat([m, m_global], dim=1)
                x = self.final_mlp(m).flatten()
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
        n_scalars_out = 1
    if obj_score:
        return TransformerModel(
            n_scalars=12,
            n_scalars_out=10,
            n_blocks=5,
            n_heads=args.n_heads,
            internal_dim=64,
            obj_score=obj_score,
            global_features_copy=args.global_features_obj_score
        )
    return TransformerModel(
        n_scalars=12,
        n_scalars_out=n_scalars_out,
        n_blocks=args.num_blocks,
        n_heads=args.n_heads,
        internal_dim=args.internal_dim,
        obj_score=obj_score
    )

