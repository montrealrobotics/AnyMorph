python3 /store/code/amorpheus_baseline/modular-rl/src/main.py --custom_xml environments/cheetah_walker_humanoids_hopper --label restricted-cheetah-walker-humanoid-hoppers-universal --expID restricted-cheetah-walker-humanoid-hoppers-universal-2 --seed 2 --lr 0.00005 --grad_clipping_value 0.1 --actor_type variational --critic_type transformer --attention_layers 3 --attention_heads 2 --attention_hidden_size 256 --transformer_norm 1 --condition_decoder 1 --use_restricted_obs --variational_frequency_encoding_size 96 --variational_latent_size 32 --variational_d_model 128 --variational_nhead 2 --variational_obs_scale 1000.0 --variational_obs_z_in_init_w 0.0 --variational_act_z_in_init_w 0.0 --variational_act_out_init_w 0.003 --variational_num_transformer_blocks 3 --variational_dim_feedforward 256 --variational_dropout 0.0 --variational_activation relu & 
python3 /store/code/amorpheus_baseline/modular-rl/src/main.py --custom_xml environments/cheetah_walker_humanoids_hopper --label restricted-cheetah-walker-humanoid-hoppers-universal --expID restricted-cheetah-walker-humanoid-hoppers-universal-3 --seed 3 --lr 0.00005 --grad_clipping_value 0.1 --actor_type variational --critic_type transformer --attention_layers 3 --attention_heads 2 --attention_hidden_size 256 --transformer_norm 1 --condition_decoder 1 --use_restricted_obs --variational_frequency_encoding_size 96 --variational_latent_size 32 --variational_d_model 128 --variational_nhead 2 --variational_obs_scale 1000.0 --variational_obs_z_in_init_w 0.0 --variational_act_z_in_init_w 0.0 --variational_act_out_init_w 0.003 --variational_num_transformer_blocks 3 --variational_dim_feedforward 256 --variational_dropout 0.0 --variational_activation relu &
wait