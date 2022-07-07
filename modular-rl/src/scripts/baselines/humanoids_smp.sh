for seed in 1 2;
do
  python3 /store/code/amorpheus_baseline/modular-rl/src/main.py \
    --custom_xml environments/humanoids \
    --seed $seed \
    --disable_fold \
    --td \
    --bu \
    --lr 0.0001 \
    --expID brandont-humanoids-smp-$seed \
    --label brandont-humanoids-smp &
done
wait
