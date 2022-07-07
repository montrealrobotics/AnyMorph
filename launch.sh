#!/bin/bash

bash launch_variational.sh --custom_xml environments/cheetahs --label cheetahs-universal-2
bash launch_variational.sh --custom_xml environments/walkers --label walkers-universal-2
bash launch_variational.sh --custom_xml environments/hoppers --label hoppers-universal-2
bash launch_variational.sh --custom_xml environments/humanoids --label humanoids-universal-2

bash launch_variational.sh --custom_xml environments/walker_humanoids --label walker-humanoids-universal-2
bash launch_variational.sh --custom_xml environments/walker_humanoids_hopper --label walker-humanoid-hoppers-universal-2
bash launch_variational.sh --custom_xml environments/cheetah_walker_humanoids --label cheetah-walker-humanoids-universal-2
bash launch_variational.sh --custom_xml environments/cheetah_walker_humanoids_hopper --label cheetah-walker-humanoid-hoppers-universal-2
