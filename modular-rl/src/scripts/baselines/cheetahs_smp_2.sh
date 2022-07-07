for seed in 0 3;
do
  python3 /store/code/amorpheus_baseline/modular-rl/src/main.py \
    --custom_xml environments/cheetahs \
    --seed $seed \
    --disable_fold \
    --td \
    --bu \
    --lr 0.0001 \
    --expID brandont-cheetahs-smp-$seed \
    --label brandont-cheetahs-smp &
done
wait

