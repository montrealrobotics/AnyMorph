for seed in 1 2;
do
  python3 /store/code/amorpheus_baseline/modular-rl/src/main.py \
    --custom_xml environments/cheetah_walker_humanoids_hopper \
    --seed $seed \
    --disable_fold \
    --td \
    --bu \
    --lr 0.0001 \
    --expID brandont-cheetah-walker-humanoid-hoppers-smp-$seed \
    --label brandont-cheetah-walker-humanoid-hoppers-smp &
done
wait

