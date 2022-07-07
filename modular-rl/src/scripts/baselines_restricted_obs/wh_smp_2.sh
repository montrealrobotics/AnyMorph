for seed in 2 3;
do
  python3 /store/code/amorpheus_baseline/modular-rl/src/main.py \
    --custom_xml environments/walker_humanoids \
    --seed $seed \
    --disable_fold \
    --td \
    --bu \
    --lr 0.0001 \
    --use_restricted_obs \
    --expID restricted-walker-humanoids-smp-$seed \
    --label restricted-walker-humanoids-smp &
done
wait
