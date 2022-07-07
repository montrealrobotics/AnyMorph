cd ../..
for seed in 1 2 3;
do
	sleep 5
        python3 main.py \
          --custom_xml environments/walker_humanoids_hopper \
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
          --label whh_amorpheus&
done
cd scripts