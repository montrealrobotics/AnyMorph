import torch

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

import math
import torch
import torch.nn as nn
from ModularActor import ActorGraphPolicy
from TransformerActor import TransformerModel
from torch.nn import TransformerDecoder, TransformerDecoderLayer
from utils import GLOBAL_SET_OF_NAMES, sinkhorn
from decoder_base import DecoderBase


torch.set_printoptions(precision=None, threshold=1e10, edgeitems=None, linewidth=None, profile=None, sci_mode=None)


class VariationalPolicy(ActorGraphPolicy):
    """a weight-sharing dynamic graph policy that changes its structure based on different morphologies and passes messages between nodes"""

    def __init__(
        self,
        state_dim,
        action_dim,
        msg_dim,
        batch_size,
        max_action,
        max_children,
        disable_fold,
        td,
        bu,
        args=None,
    ):
        super(ActorGraphPolicy, self).__init__()
        self.args = args
        self.num_limbs = 1
        self.msg_down = [None] * self.num_limbs
        self.msg_up = [None] * self.num_limbs
        self.action = [None] * self.num_limbs
        self.input_state = [None] * self.num_limbs
        self.max_action = max_action
        self.msg_dim = msg_dim
        self.batch_size = batch_size
        self.max_children = max_children
        self.disable_fold = disable_fold
        self.state_dim = state_dim
        self.action_dim = action_dim
        self.latent_size = args.variational_latent_size

        self.actor = DecoderBase(
            frequency_encoding_size=args.variational_frequency_encoding_size, 
            latent_size=args.variational_latent_size, 
            d_model=args.variational_d_model, 
            nhead=args.variational_nhead,
            obs_scale=args.variational_obs_scale, 

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

    def forward(self, state, mode="train"):
        
        self.clear_buffer()

        if mode == "inference":
            temp, self.batch_size = self.batch_size, 1

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

        self.action = self.actor(obs_embedding, act_embedding, state.permute(1, 0))
        self.action = self.max_action * torch.tanh(self.action)

        self.action = self.action.permute(1, 0)

        if mode == "inference":
            self.batch_size = temp

        return self.action

    def change_morphology(self, parents, action_ids):

        self.parents = parents
        self.action_ids = torch.LongTensor(action_ids).to(device)
        self.num_limbs = len(parents)
        self.msg_down = [None] * self.num_limbs
        self.msg_up = [None] * self.num_limbs
        self.action = [None] * self.num_limbs
        self.input_state = [None] * self.num_limbs


class VariationalPolicy2(ActorGraphPolicy):
    """a weight-sharing dynamic graph policy that changes its structure based on different morphologies and passes messages between nodes"""

    def __init__(
        self,
        state_dim,
        action_dim,
        msg_dim,
        batch_size,
        max_action,
        max_children,
        disable_fold,
        td,
        bu,
        args=None,
    ):
        super(ActorGraphPolicy, self).__init__()
        self.args = args
        self.num_limbs = 1
        self.msg_down = [None] * self.num_limbs
        self.msg_up = [None] * self.num_limbs
        self.action = [None] * self.num_limbs
        self.input_state = [None] * self.num_limbs
        self.max_action = max_action
        self.msg_dim = msg_dim
        self.batch_size = batch_size
        self.max_children = max_children
        self.disable_fold = disable_fold
        self.state_dim = state_dim
        self.action_dim = action_dim

        self.actor = TransformerModel(
            self.state_dim,
            action_dim,
            args.attention_embedding_size,
            args.attention_heads,
            args.attention_hidden_size,
            args.attention_layers,
            args.dropout_rate,
            condition_decoder=args.condition_decoder_on_features,
            transformer_norm=args.transformer_norm).to(device)

        self.perm_left_embedding = nn.Embedding(
            self.state_dim * 
            len(GLOBAL_SET_OF_NAMES), 
            args.attention_embedding_size).to(device)

        self.perm_right_embedding = nn.Embedding(
            self.state_dim * 
            len(GLOBAL_SET_OF_NAMES), 
            args.attention_embedding_size).to(device)

        self.permutation_weight = nn.Parameter(torch.empty((
            len(GLOBAL_SET_OF_NAMES),
            args.attention_embedding_size, 
            args.attention_embedding_size)).to(device))

        self.permutation_weight.data.uniform_(-0.01, 0.01)

    def get_amorpheus_perm_slice(self):

        perm_ids = torch.arange(self.num_limbs * self.state_dim, 
                                dtype=torch.int64, device=device)

        perm_ids = torch.stack([perm_ids.roll(
            -self.state_dim * limb_id) for limb_id in range(self.num_limbs)])

        perm = nn.functional.one_hot(
            perm_ids, self.state_dim * self.num_limbs).to(torch.float32)

        return perm.view(
            self.num_limbs, 1, 
            self.state_dim * self.num_limbs, 
            self.state_dim * self.num_limbs)

    def get_learned_perm_slice(self):

        obs_ids = torch.repeat_interleave(
            self.action_ids, self.state_dim) * self.state_dim + torch.arange(
                self.state_dim, 
                dtype=torch.int64, device=device).repeat(self.num_limbs)

        perm_left_embeddings = self.perm_left_embedding(obs_ids).view(
            1, 1, self.state_dim * self.num_limbs, 
            self.args.attention_embedding_size).contiguous()

        perm_right_embeddings = self.perm_right_embedding(obs_ids).view(
            1, 1, self.state_dim * self.num_limbs, 
            self.args.attention_embedding_size).contiguous()

        perm_weights = nn.functional.embedding(
            self.action_ids, 
            self.permutation_weight.view(len(GLOBAL_SET_OF_NAMES), -1)).view(
                self.num_limbs, 1, 
                self.args.attention_embedding_size, 
                self.args.attention_embedding_size).contiguous()

        return sinkhorn(perm_left_embeddings @ 
                        perm_weights @ 
                        perm_right_embeddings.permute(0, 1, 3, 2)).exp()

    def forward(self, state, mode="train"):
        
        self.clear_buffer()

        if mode == "inference":
            temp, self.batch_size = self.batch_size, 1

        state = state.view(1, self.batch_size, self.state_dim * self.num_limbs, 1)

        perm = self.get_learned_perm_slice()
        self.perm_loss = ((perm - self.get_amorpheus_perm_slice()) ** 2).mean()

        self.input_state = torch.matmul(perm, state)[:, :, :self.state_dim, 0]

        self.action = self.actor(self.input_state)
        self.action = self.max_action * torch.tanh(self.action)

        # because of the permutation of the states, we need to 
        # unpermute the actions now so that the actions are (batch,actions)
        self.action = self.action.permute(1, 0, 2)

        if mode == "inference":
            self.batch_size = temp

        return torch.squeeze(self.action)

    def change_morphology(self, parents, action_ids):
        self.parents = parents
        self.action_ids = torch.LongTensor(action_ids).to(device)
        self.num_limbs = len(parents)
        self.msg_down = [None] * self.num_limbs
        self.msg_up = [None] * self.num_limbs
        self.action = [None] * self.num_limbs
        self.input_state = [None] * self.num_limbs
