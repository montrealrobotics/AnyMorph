for seed in 0 1;
do
  python3 /store/code/amorpheus_baseline/modular-rl/src/main.py \
    --custom_xml environments/walker_humanoids_hopper \
    --seed $seed \
    --disable_fold \
    --td \
    --bu \
    --lr 0.0001 \
    --use_restricted_obs \
    --expID restricted-walker-humanoid-hoppers-smp-$seed \
    --label restricted-walker-humanoid-hoppers-smp &
done
wait
