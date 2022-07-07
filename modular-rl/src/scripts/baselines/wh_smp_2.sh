for seed in 0 3;
do
  python3 /store/code/amorpheus_baseline/modular-rl/src/main.py \
    --custom_xml environments/walker_humanoids \
    --seed $seed \
    --disable_fold \
    --td \
    --bu \
    --lr 0.0001 \
    --expID brandont-walker-humanoids-smp-$seed \
    --label brandont-walker-humanoids-smp &
done
wait
