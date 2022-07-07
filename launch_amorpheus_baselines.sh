#!/bin/bash

for morphologies in cheetahs walkers hoppers humanoids wh whh cwh cwhh;
do

  make_job_template.py \
    --pytorch \
    --name brandont-${morphologies}-amorpheus \
    --image amr-registry.caas.intel.com/aipg/brandont-amorpheus-baseline:latest \
    --gpus 1 \
    --cpus 8 \
    --memory 32 \
    --command "bash /store/code/amorpheus_baseline/modular-rl/src/scripts/baselines/${morphologies}_amorpheus.sh" \
    --run

  make_job_template.py \
    --pytorch \
    --name brandont-${morphologies}-amorpheus \
    --image amr-registry.caas.intel.com/aipg/brandont-amorpheus-baseline:latest \
    --gpus 1 \
    --cpus 8 \
    --memory 32 \
    --command "bash /store/code/amorpheus_baseline/modular-rl/src/scripts/baselines/${morphologies}_amorpheus_2.sh" \
    --run

done