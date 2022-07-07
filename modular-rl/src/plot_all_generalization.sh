#!/bin/bash

for seed in 0 1 2 3;
do

    python3 plot_generalization.py --expID cheetahs-universal-2-$seed --custom_xml environments/cheetahs --custom_xml_held_out environments/cheetahs_test
    python3 plot_generalization.py --expID brandont-cheetahs-smp-$seed --custom_xml environments/cheetahs --custom_xml_held_out environments/cheetahs_test
    python3 plot_generalization.py --expID brandont-cheetahs-amorpheus-$seed --custom_xml environments/cheetahs --custom_xml_held_out environments/cheetahs_test
    
    python3 plot_generalization.py --expID walkers-universal-2-$seed --custom_xml environments/walkers --custom_xml_held_out environments/walkers_test
    python3 plot_generalization.py --expID brandont-walkers-smp-$seed --custom_xml environments/walkers --custom_xml_held_out environments/walkers_test
    python3 plot_generalization.py --expID brandont-walkers-amorpheus-$seed --custom_xml environments/walkers --custom_xml_held_out environments/walkers_test
    
    python3 plot_generalization.py --expID humanoids-universal-2-$seed --custom_xml environments/humanoids --custom_xml_held_out environments/humanoids_test
    python3 plot_generalization.py --expID brandont-humanoids-smp-$seed --custom_xml environments/humanoids --custom_xml_held_out environments/humanoids_test
    python3 plot_generalization.py --expID brandont-humanoids-amorpheus-$seed --custom_xml environments/humanoids --custom_xml_held_out environments/humanoids_test
    
    
    python3 plot_generalization.py --expID walker-humanoids-universal-2-$seed --custom_xml environments/walker_humanoids --custom_xml_held_out environments/walker_humanoids_test
    python3 plot_generalization.py --expID brandont-walker-humanoids-smp-$seed --custom_xml environments/walker_humanoids --custom_xml_held_out environments/walker_humanoids_test
    python3 plot_generalization.py --expID brandont-walker-humanoids-amorpheus-$seed --custom_xml environments/walker_humanoids --custom_xml_held_out environments/walker_humanoids_test
    
    
    python3 plot_generalization.py --expID walker-humanoid-hoppers-universal-2-$seed --custom_xml environments/walker_humanoids_hopper --custom_xml_held_out environments/walker_humanoids_hopper_test
    python3 plot_generalization.py --expID brandont-walker-humanoid-hoppers-smp-$seed --custom_xml environments/walker_humanoids_hopper --custom_xml_held_out environments/walker_humanoids_hopper_test
    python3 plot_generalization.py --expID brandont-walker-humanoid-hoppers-amorpheus-$seed --custom_xml environments/walker_humanoids_hopper --custom_xml_held_out environments/walker_humanoids_hopper_test
    
    
    python3 plot_generalization.py --expID cheetah-walker-humanoids-universal-2-$seed --custom_xml environments/cheetah_walker_humanoids --custom_xml_held_out environments/cheetah_walker_humanoids_test
    python3 plot_generalization.py --expID brandont-cheetah-walker-humanoids-smp-$seed --custom_xml environments/cheetah_walker_humanoids --custom_xml_held_out environments/cheetah_walker_humanoids_test
    python3 plot_generalization.py --expID brandont-cheetah-walker-humanoids-amorpheus-$seed --custom_xml environments/cheetah_walker_humanoids --custom_xml_held_out environments/cheetah_walker_humanoids_test
    
    
    python3 plot_generalization.py --expID cheetah-walker-humanoid-hoppers-universal-2-$seed --custom_xml environments/cheetah_walker_humanoids_hopper --custom_xml_held_out environments/cheetah_walker_humanoids_hopper_test
    python3 plot_generalization.py --expID brandont-cheetah-walker-humanoid-hoppers-smp-$seed --custom_xml environments/cheetah_walker_humanoids_hopper --custom_xml_held_out environments/cheetah_walker_humanoids_hopper_test
    python3 plot_generalization.py --expID brandont-cheetah-walker-humanoid-hoppers-amorpheus-$seed --custom_xml environments/cheetah_walker_humanoids_hopper --custom_xml_held_out environments/cheetah_walker_humanoids_hopper_test

done

