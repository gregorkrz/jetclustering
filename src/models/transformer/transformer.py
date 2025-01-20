from src.models.transformer.tr_blocks import Transformer
import torch
import torch.nn as nn
from xformers.ops.fmha import BlockDiagonalMask

class TransformerModel(torch.nn.Module):
    def __init__(self, n_scalars, n_scalars_out):
        super().__init__()
        self.n_scalars = n_scalars
        self.input_dim = n_scalars + 3
        self.output_dim = 3
        internal_dim = 128
        #self.custom_decoder = nn.Linear(internal_dim, self.output_dim)
        n_heads = 4
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
        self.transformer = Transformer(
            in_channels=self.input_dim,
            out_channels=self.output_dim,
            hidden_channels=internal_dim,
            num_heads=n_heads,
            num_blocks=10,
        )
        self.batch_norm = nn.BatchNorm1d(self.input_dim, momentum=0.1)
        #self.clustering = nn.Linear(3, self.output_dim - 1, bias=False)

    def forward(self, data):
        # data: instance of EventBatch
        inputs_v = data.input_vectors
        inputs_scalar = data.input_scalars
        assert inputs_scalar.shape[1] == self.n_scalars, "Expected %d, got %d" % (self.n_scalars, inputs_scalar.shape[1])
        inputs_transformer = torch.cat([inputs_scalar, inputs_v], dim=1)
        assert inputs_transformer.shape[1] == self.input_dim
        mask = self.build_attention_mask(data.batch_idx)
        x = self.batch_norm(inputs_transformer).unsqueeze(0)
        # convert inputs to float16
        x = self.transformer(x, attention_mask=mask)[0]
        #x = self.custom_decoder(x)
        assert x.shape[1] == self.output_dim, "Expected %d, got %d" % (self.output_dim, x.shape[1])
        assert x.shape[0] == inputs_transformer.shape[0], "Expected %d, got %d" % (inputs_transformer.shape[0], x.shape[0])
        x[:, -1] = torch.sigmoid(x[:, -1])
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
        n_scalars_out = 1
    return TransformerModel(
        n_scalars=12,
        n_scalars_out=n_scalars_out
    )
