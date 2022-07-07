#!/bin/bash


seed=${seed:-0}
morphologies=${morphologies:-"walkers"}

while [ $# -gt 0 ]; do
  if [[ $1 == *"--"* ]]; then
    param="${1/--/}"; declare $param="$2"
  fi; shift
done

morphologies_label=${morphologies//humanoid-hoppers/humanoids-hopper}

cd /store/code/amorpheus_baseline/modular-rl/src

python3 plot_attention.py --expID ${morphologies}-universal-2-${seed} --custom_xml environments/${morphologies_label//-/_}
python3 plot_attention.py --expID brandont-${morphologies}-amorpheus-${seed} --custom_xml environments/${morphologies_label//-/_}
python3 plot_attention.py --expID brandont-${morphologies}-smp-${seed} --custom_xml environments/${morphologies_label//-/_}
