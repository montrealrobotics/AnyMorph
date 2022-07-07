#!/bin/bash

for morphologies in cheetahs walkers hoppers humanoids walker-humanoids walker-humanoid-hoppers cheetah-walker-humanoids cheetah-walker-humanoid-hoppers;
do

for seed in 0 1 2 3;
do

  make_job_template.py \
    --pytorch \
    --name corruptions-${morphologies}-${seed} \
    --image amr-registry.caas.intel.com/aipg/brandont-amorpheus-baseline:latest \
    --gpus 1 \
    --cpus 8 \
    --memory 32 \
    --command "bash /store/code/amorpheus_baseline/modular-rl/src/scripts/corruptions/launch.sh --seed ${seed} --morphologies ${morphologies}" \
    --run

done
done