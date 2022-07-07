import math
import torch
import torch.nn as nn


class DecoderBase(nn.Module):

    def __init__(self, frequency_encoding_size: int = 96, latent_size: int = 32, 
                 d_model: int = 128, nhead: int = 4, is_critic: bool = False, 
                 obs_scale: float = 1000.0, act_scale: float = 1000.0, 
                 obs_z_in_init_w: float = 0.0, act_z_in_init_w: float = 0.0, 
                 act_out_init_w: float = 3e-3, 
                 num_transformer_blocks: int = 3, dim_feedforward: int = 256, 
                 dropout: float = 0.0, activation: str = "relu"):

        super(DecoderBase, self).__init__()

        self.frequency_encoding_size = frequency_encoding_size
        self.latent_size = latent_size
        self.d_model = d_model
        self.nhead = nhead
        self.obs_scale = obs_scale
        self.act_scale = act_scale
        self.num_transformer_blocks = num_transformer_blocks
        self.is_critic = is_critic

        self.obs_z_input_layer = nn.Linear(
            latent_size + frequency_encoding_size, d_model)
        if obs_z_in_init_w > 0:
            self.obs_z_input_layer.weight.data\
                .uniform_(-obs_z_in_init_w, obs_z_in_init_w)

        self.act_z_input_layer = nn.Linear(latent_size + (
            frequency_encoding_size if is_critic else 0), d_model)
        if act_z_in_init_w > 0:
            self.act_z_input_layer.weight.data\
                .uniform_(-act_z_in_init_w, act_z_in_init_w)

        self.act_output_layer = nn.Linear(d_model + latent_size, 1)
        self.act_output_layer.bias.data.fill_(0)
        if act_out_init_w > 0:
            self.act_output_layer.weight.data\
                .uniform_(-act_out_init_w, act_out_init_w)

        self.obs_norm = nn.LayerNorm(d_model)
        self.act_norm = nn.LayerNorm(d_model)

        self.transformer = nn.Transformer(
            d_model=d_model, dim_feedforward=dim_feedforward, 
            nhead=nhead, dropout=dropout, activation=activation,
            num_encoder_layers=num_transformer_blocks,
            num_decoder_layers=num_transformer_blocks)

    @staticmethod
    def frequency_encoding(x, d_model):

        idx = torch.arange(0, d_model, 2).to(dtype=x.dtype, device=x.device)
        div_term = torch.exp(idx * (-math.log(10000.0) / d_model))

        x = x.unsqueeze(-1)
        while len(div_term.shape) < len(x.shape):
            div_term = div_term.unsqueeze(0)

        return torch.cat([torch.sin(x * div_term),
                          torch.cos(x * div_term)], dim=-1)

    def forward(self, obs_z, act_z, obs, act=None):

        obs = (2 / math.sqrt(2.0)) * self.frequency_encoding(
            obs * self.obs_scale, self.frequency_encoding_size)

        n_obs_z = torch.cat([obs_z, obs], dim=2)

        if self.is_critic:

            assert act is not None, \
                "q function must condition on the action"

            act = (2 / math.sqrt(2.0)) * self.frequency_encoding(
                act * self.act_scale, self.frequency_encoding_size)

            n_act_z = torch.cat([act_z, act], dim=2)

        else:
            
            n_act_z = act_z

        n_obs_z = self.obs_z_input_layer(n_obs_z)
        n_act_z = self.act_z_input_layer(n_act_z)

        act = self.transformer(self.obs_norm(n_obs_z), self.act_norm(n_act_z))

        return self.act_output_layer(
            torch.cat([act_z, act], dim=2)).squeeze(2)
