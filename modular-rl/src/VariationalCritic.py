from __future__ import print_function
import torch
import torch.nn as nn

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
from TransformerActor import TransformerModel
from utils import GLOBAL_SET_OF_NAMES, sinkhorn
from decoder_base import DecoderBase


class CriticVariationalPolicy(nn.Module):
    """a weight-sharing dynamic graph policy that changes its structure based on different morphologies and passes messages between nodes"""

    def __init__(
        self,
        state_dim,
        action_dim,
        msg_dim,
        batch_size,
        max_children,
        disable_fold,
        td,
        bu,
        args=None,
    ):
        super().__init__()
        self.num_limbs = 1
        self.x1 = [None] * self.num_limbs
        self.x2 = [None] * self.num_limbs
        self.input_state = [None] * self.num_limbs
        self.input_action = [None] * self.num_limbs
        self.msg_down = [None] * self.num_limbs
        self.msg_up = [None] * self.num_limbs
        self.msg_dim = msg_dim
        self.batch_size = batch_size
        self.max_children = max_children
        self.disable_fold = disable_fold
        self.state_dim = state_dim
        self.action_dim = action_dim
        self.latent_size = args.variational_latent_size

        self.critic1 = DecoderBase(
            frequency_encoding_size=args.variational_frequency_encoding_size, 
            latent_size=args.variational_latent_size, 
            d_model=args.variational_d_model, 
            nhead=args.variational_nhead,
            obs_scale=args.variational_obs_scale, 
            act_scale=args.variational_act_scale, 

            is_critic=True,

            obs_z_in_init_w=args.variational_obs_z_in_init_w, 
            act_z_in_init_w=args.variational_act_z_in_init_w, 
            act_out_init_w=args.variational_act_out_init_w, 

            num_transformer_blocks=args.variational_num_transformer_blocks,
            dim_feedforward=args.variational_dim_feedforward, 
            
            dropout=args.variational_dropout,
            activation=args.variational_activation,
            ).to(device)

        self.critic2 = DecoderBase(
            frequency_encoding_size=args.variational_frequency_encoding_size, 
            latent_size=args.variational_latent_size, 
            d_model=args.variational_d_model, 
            nhead=args.variational_nhead,
            obs_scale=args.variational_obs_scale, 
            act_scale=args.variational_act_scale, 

            is_critic=True,

            obs_z_in_init_w=args.variational_obs_z_in_init_w, 
            act_z_in_init_w=args.variational_act_z_in_init_w, 
            act_out_init_w=args.variational_act_out_init_w, 

            num_transformer_blocks=args.variational_num_transformer_blocks,
            dim_feedforward=args.variational_dim_feedforward, 
            
            dropout=args.variational_dropout,
            activation=args.variational_activation,
            ).to(device)

        self.obs_embeddings = nn.Embedding(
            self.state_dim * len(GLOBAL_SET_OF_NAMES), self.latent_size).to(device)

        self.act_embeddings = nn.Embedding(
            len(GLOBAL_SET_OF_NAMES), self.latent_size).to(device)

    def forward(self, state, action):

        self.clear_buffer()

        assert (
            state.shape[1] == self.state_dim * self.num_limbs
        ), "state.shape[1] expects {} but got {} with num_limbs being {} and state_dim being {}".format(
            self.state_dim * self.num_limbs,
            state.shape[1],
            self.num_limbs,
            self.state_dim,
        )

        obs_ids = torch.repeat_interleave(
            self.action_ids, self.state_dim) * self.state_dim + torch.arange(
                self.state_dim, 
                dtype=torch.int64, device=device).repeat(self.num_limbs)

        obs_embedding = self.obs_embeddings(obs_ids)
        act_embedding = self.act_embeddings(self.action_ids)

        obs_embedding = obs_embedding.view(
            self.state_dim * self.num_limbs, 1, self.latent_size)
        obs_embedding = obs_embedding.expand(
            self.state_dim * self.num_limbs, self.batch_size, self.latent_size)

        act_embedding = act_embedding.view(
            self.num_limbs, 1, self.latent_size)
        act_embedding = act_embedding.expand(
            self.num_limbs, self.batch_size, self.latent_size)

        self.x1 = self.critic1(obs_embedding, act_embedding, 
                               state.permute(1, 0), 
                               act=action.permute(1, 0)).permute(1, 0)

        self.x2 = self.critic2(obs_embedding, act_embedding, 
                               state.permute(1, 0), 
                               act=action.permute(1, 0)).permute(1, 0)

        return self.x1, self.x2

    def Q1(self, state, action):

        self.clear_buffer()

        obs_ids = torch.repeat_interleave(
            self.action_ids, self.state_dim) * self.state_dim + torch.arange(
                self.state_dim, 
                dtype=torch.int64, device=device).repeat(self.num_limbs)

        obs_embedding = self.obs_embeddings(obs_ids)
        act_embedding = self.act_embeddings(self.action_ids)

        obs_embedding = obs_embedding.view(
            self.state_dim * self.num_limbs, 1, self.latent_size)
        obs_embedding = obs_embedding.expand(
            self.state_dim * self.num_limbs, self.batch_size, self.latent_size)

        act_embedding = act_embedding.view(
            self.num_limbs, 1, self.latent_size)
        act_embedding = act_embedding.expand(
            self.num_limbs, self.batch_size, self.latent_size)

        self.x1 = self.critic1(obs_embedding, act_embedding, 
                               state.permute(1, 0), 
                               act=action.permute(1, 0)).permute(1, 0)

        return self.x1

    def clear_buffer(self):
        self.x1 = [None] * self.num_limbs
        self.x2 = [None] * self.num_limbs
        self.input_state = [None] * self.num_limbs
        self.input_action = [None] * self.num_limbs
        self.msg_down = [None] * self.num_limbs
        self.msg_up = [None] * self.num_limbs
        self.zeroFold_td = None
        self.zeroFold_bu = None
        self.fold = None

    def change_morphology(self, parents, action_ids):
        self.parents = parents
        self.action_ids = torch.LongTensor(action_ids).to(device)
        self.num_limbs = len(parents)
        self.msg_down = [None] * self.num_limbs
        self.msg_up = [None] * self.num_limbs
        self.action = [None] * self.num_limbs
        self.input_state = [None] * self.num_limbs
