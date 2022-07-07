#!/bin/bash

custom_xml=${custom_xml:-'environments/cheetah_walker_humanoids'}
label=${label:-"cheetah-walker-humanoids-universal"}
grad_clipping_value=${grad_clipping_value:-0.1}
lr=${lr:-0.00005}

variational_frequency_encoding_size=${variational_frequency_encoding_size:-96}
variational_latent_size=${variational_latent_size:-32}
variational_d_model=${variational_d_model:-128}
variational_nhead=${variational_nhead:-2}

variational_obs_scale=${variational_obs_scale:-1000.0}
variational_act_out_init_w=${variational_act_out_init_w:-0.003}
variational_obs_z_in_init_w=${variational_obs_z_in_init_w:-0.0}
variational_act_z_in_init_w=${variational_act_z_in_init_w:-0.0}

variational_num_transformer_blocks=${variational_num_transformer_blocks:-3}
variational_dim_feedforward=${variational_dim_feedforward:-256}
variational_dropout=${variational_dropout:-0.0}
variational_activation=${variational_activation:-'relu'}

while [ $# -gt 0 ]; do
  if [[ $1 == *"--"* ]]; then
    param="${1/--/}"; declare $param="$2"
  fi; shift
done

seed=0

COMMAND_STR="python3 /store/code/amorpheus_baseline/modular-rl/src/main.py \
--custom_xml ${custom_xml} \
--label ${label} \
--expID ${label}-${seed} \
--seed ${seed} \
--lr ${lr} \
--grad_clipping_value ${grad_clipping_value} \
--actor_type variational \
--critic_type transformer \
--attention_layers 3 \
--attention_heads 2 \
--attention_hidden_size 256 \
--transformer_norm 1 \
--condition_decoder 1 \
--variational_frequency_encoding_size ${variational_frequency_encoding_size} \
--variational_latent_size ${variational_latent_size} \
--variational_d_model ${variational_d_model} \
--variational_nhead ${variational_nhead} \
--variational_obs_scale ${variational_obs_scale} \
--variational_obs_z_in_init_w ${variational_obs_z_in_init_w} \
--variational_act_z_in_init_w ${variational_act_z_in_init_w} \
--variational_act_out_init_w ${variational_act_out_init_w} \
--variational_num_transformer_blocks ${variational_num_transformer_blocks} \
--variational_dim_feedforward ${variational_dim_feedforward} \
--variational_dropout ${variational_dropout} \
--variational_activation ${variational_activation}"

echo "$COMMAND_STR"

make_job_template.py \
  --pytorch \
  --name brandont-${label}-${seed} \
  --image amr-registry.caas.intel.com/aipg/brandont-amorpheus-baseline:latest \
  --gpus 1 \
  --cpus 8 \
  --memory 32 \
  --command "$COMMAND_STR" \
  --run

seed=1

COMMAND_STR="python3 /store/code/amorpheus_baseline/modular-rl/src/main.py \
--custom_xml ${custom_xml} \
--label ${label} \
--expID ${label}-${seed} \
--seed ${seed} \
--lr ${lr} \
--grad_clipping_value ${grad_clipping_value} \
--actor_type variational \
--critic_type transformer \
--attention_layers 3 \
--attention_heads 2 \
--attention_hidden_size 256 \
--transformer_norm 1 \
--condition_decoder 1 \
--variational_frequency_encoding_size ${variational_frequency_encoding_size} \
--variational_latent_size ${variational_latent_size} \
--variational_d_model ${variational_d_model} \
--variational_nhead ${variational_nhead} \
--variational_obs_scale ${variational_obs_scale} \
--variational_obs_z_in_init_w ${variational_obs_z_in_init_w} \
--variational_act_z_in_init_w ${variational_act_z_in_init_w} \
--variational_act_out_init_w ${variational_act_out_init_w} \
--variational_num_transformer_blocks ${variational_num_transformer_blocks} \
--variational_dim_feedforward ${variational_dim_feedforward} \
--variational_dropout ${variational_dropout} \
--variational_activation ${variational_activation}"

echo "$COMMAND_STR"

make_job_template.py \
  --pytorch \
  --name brandont-${label}-${seed} \
  --image amr-registry.caas.intel.com/aipg/brandont-amorpheus-baseline:latest \
  --gpus 1 \
  --cpus 8 \
  --memory 32 \
  --command "$COMMAND_STR" \
  --run

seed=2

COMMAND_STR="python3 /store/code/amorpheus_baseline/modular-rl/src/main.py \
--custom_xml ${custom_xml} \
--label ${label} \
--expID ${label}-${seed} \
--seed ${seed} \
--lr ${lr} \
--grad_clipping_value ${grad_clipping_value} \
--actor_type variational \
--critic_type transformer \
--attention_layers 3 \
--attention_heads 2 \
--attention_hidden_size 256 \
--transformer_norm 1 \
--condition_decoder 1 \
--variational_frequency_encoding_size ${variational_frequency_encoding_size} \
--variational_latent_size ${variational_latent_size} \
--variational_d_model ${variational_d_model} \
--variational_nhead ${variational_nhead} \
--variational_obs_scale ${variational_obs_scale} \
--variational_obs_z_in_init_w ${variational_obs_z_in_init_w} \
--variational_act_z_in_init_w ${variational_act_z_in_init_w} \
--variational_act_out_init_w ${variational_act_out_init_w} \
--variational_num_transformer_blocks ${variational_num_transformer_blocks} \
--variational_dim_feedforward ${variational_dim_feedforward} \
--variational_dropout ${variational_dropout} \
--variational_activation ${variational_activation}"

echo "$COMMAND_STR"

make_job_template.py \
  --pytorch \
  --name brandont-${label}-${seed} \
  --image amr-registry.caas.intel.com/aipg/brandont-amorpheus-baseline:latest \
  --gpus 1 \
  --cpus 8 \
  --memory 32 \
  --command "$COMMAND_STR" \
  --run

seed=3

COMMAND_STR="python3 /store/code/amorpheus_baseline/modular-rl/src/main.py \
--custom_xml ${custom_xml} \
--label ${label} \
--expID ${label}-${seed} \
--seed ${seed} \
--lr ${lr} \
--grad_clipping_value ${grad_clipping_value} \
--actor_type variational \
--critic_type transformer \
--attention_layers 3 \
--attention_heads 2 \
--attention_hidden_size 256 \
--transformer_norm 1 \
--condition_decoder 1 \
--variational_frequency_encoding_size ${variational_frequency_encoding_size} \
--variational_latent_size ${variational_latent_size} \
--variational_d_model ${variational_d_model} \
--variational_nhead ${variational_nhead} \
--variational_obs_scale ${variational_obs_scale} \
--variational_obs_z_in_init_w ${variational_obs_z_in_init_w} \
--variational_act_z_in_init_w ${variational_act_z_in_init_w} \
--variational_act_out_init_w ${variational_act_out_init_w} \
--variational_num_transformer_blocks ${variational_num_transformer_blocks} \
--variational_dim_feedforward ${variational_dim_feedforward} \
--variational_dropout ${variational_dropout} \
--variational_activation ${variational_activation}"

echo "$COMMAND_STR"

make_job_template.py \
  --pytorch \
  --name brandont-${label}-${seed} \
  --image amr-registry.caas.intel.com/aipg/brandont-amorpheus-baseline:latest \
  --gpus 1 \
  --cpus 8 \
  --memory 32 \
  --command "$COMMAND_STR" \
  --run
