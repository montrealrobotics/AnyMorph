#!/bin/bash

lr=0.00001
variational_act_out_init_w=0.003


for morphologies in walker_humanoids walker_humanoids_hopper cheetah_walker_humanoids cheetah_walker_humanoids_hopper;
do

  morphologies_label=${morphologies/humanoids_hopper/humanoid_hoppers}

  for seed in 0 1 2 3;
  do

    command_str="python3 /store/code/amorpheus_baseline/modular-rl/src/main.py \
--custom_xml environments/${morphologies} \
--use_restricted_obs \
--label restricted-${morphologies_label//_/-}-universal \
--expID restricted-${morphologies_label//_/-}-full-2-universal-${seed} \
--seed ${seed} \
--lr ${lr} \
--grad_clipping_value 0.1 \
--actor_type variational \
--critic_type variational \
--attention_layers 3 \
--attention_heads 2 \
--attention_hidden_size 256 \
--transformer_norm 1 \
--condition_decoder 1 \
--variational_frequency_encoding_size 96 \
--variational_latent_size 32 \
--variational_d_model 128 \
--variational_nhead 2 \
--variational_obs_scale 1000.0 \
--variational_act_scale 1000.0 \
--variational_obs_z_in_init_w 0.0 \
--variational_act_z_in_init_w 0.0 \
--variational_act_out_init_w ${variational_act_out_init_w} \
--variational_num_transformer_blocks 3 \
--variational_dim_feedforward 256 \
--variational_dropout 0.0 \
--variational_activation relu"

    echo "$command_str"

    make_job_template.py \
      --pytorch \
      --name restricted-${morphologies//_/-}-universal-full-2 \
      --image amr-registry.caas.intel.com/aipg/brandont-amorpheus-baseline:latest \
      --gpus 1 \
      --cpus 8 \
      --memory 32 \
      --command "$command_str" \
      --run

    done

done