for seed in 1 2;
do
  python3 /store/code/amorpheus_baseline/modular-rl/src/main.py \
    --custom_xml environments/cheetahs \
    --actor_type transformer \
    --critic_type transformer \
    --seed $seed \
    --grad_clipping_value 0.1 \
    --attention_layers 3 \
    --attention_heads 2 \
    --lr 0.0001 \
    --transformer_norm 1 \
    --attention_hidden_size 256 \
    --condition_decoder 1 \
    --expID brandont-cheetahs-amorpheus-$seed \
    --label brandont-cheetahs-amorpheus &
done
wait