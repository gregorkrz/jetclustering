import torch

class IdentityModel(torch.nn.Module):
    def __init__(self, n_out_coords=3):
        super().__init__()
        self.n_out_coords = n_out_coords

    def forward(self, data):
        # data: instance of EventBatch
        inputs_v = data.input_vectors # four-momenta
        betas = torch.ones(data.input_vectors.shape[0]).to(inputs_v.device)
        norm_inputs_v = torch.norm(inputs_v, dim=1).unsqueeze(1)
        #print("inputs_v.shape", inputs_v.shape)
        #print("betas.shape", betas.shape)
        #print("norm_inputs_v.shape", norm_inputs_v.shape)
        #print("betas unsqueezed shape", betas.unsqueeze(1).shape)
        x = torch.cat([inputs_v / norm_inputs_v, betas.unsqueeze(1)], dim=1)
        return x


def get_model(args):
    return IdentityModel()
