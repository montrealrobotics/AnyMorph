for seed in 2 3;
do
  python3 /store/code/amorpheus_baseline/modular-rl/src/main.py \
    --custom_xml environments/cheetah_walker_humanoids \
    --seed $seed \
    --disable_fold \
    --td \
    --bu \
    --lr 0.0001 \
    --use_restricted_obs \
    --expID restricted-cheetah-walker-humanoids-smp-$seed \
    --label restricted-cheetah-walker-humanoids-smp &
done
wait

