## AnyMorph: Learning Transferable Polices By Inferring Agent Morphology
### ICML 2022
#### [OpenReview](https://proceedings.mlr.press/v162/trabucco22b/trabucco22b.pdf) | [Arxiv](https://arxiv.org/abs/2206.12279)
#### [Blog post](https://www.linkedin.com/pulse/anymorph-learning-transferable-policies-inferring-agent-trabucco/) and [here as Mila](https://mila.quebec/en/article/anymorph-learning-transferable-policies-by-inferring-agent-morphology/)

[Brandon Trabucco](https://twitter.com/brandontrabucco), [Mariano Phielipp](https://twitter.com/mphielipp), [Glen Berseth](https://twitter.com/GlenBerseth)



### TL;DR 

Our paper, Learning Transferable Policies By Inferring Agent Morphology, delivers state-of-the-art generalization and robustness for controlling large collections of reinforcement learning agents with diverse morphologies and designs.

```
@inproceedings{Trabucco2022AnyMorph,
title={AnyMorph: Learning Transferable Policies By Inferring Agent Morphology},
author={Trabucco Brandon and Phielipp Mariano and Glen Berseth},
journal={International Conference on Machine Learning},
year={2022}
}
```

![](https://mila.quebec/wp-content/uploads/2022/08/anymorph_prompt-1.gif)

## Setup

All the experiments are done in a Docker container.
To build it, run `./docker_build.sh <device>`, where `<device>` can be `cpu` or `cu101`. It will use CUDA by default.

To build and run the experiments, you need a MuJoCo license. Put it to the root folder before running `docker_build.sh`. 


## Running

```
./docker_run <device_id> # either GPU id or cpu
cd amorpheus             # select the experiment to replicate
bash cwhh.sh             # run it on a task
```

We were using [Sacred](https://github.com/IDSIA/sacred) with a remote MongoDB for experiment management.
For release, we changed Sacred to log to local files instead.
You can change it back to MongDB if you provide credentials in `modular-rl/src/main.py`. 

## Acknowledgement

- The code is built on top of [SMP](https://github.com/huangwl18/modular-rl) repository. 
- NerveNet Walkers environment are
taken and adapted from [the original repo](https://github.com/WilsonWangTHU/NerveNet).
- Initial implementation of the transformers was taken from the [official Pytorch tutorial](https://pytorch.org/tutorials/beginner/transformer_tutorial.html) and modified thereafter. 
