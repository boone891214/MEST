# MEST: Accurate and Fast Memory-Economic Sparse Training Framework on the Edge
Sample code used for NeurIPS 2021 paper:
[MEST: Accurate and Fast Memory-Economic Sparse Training Framework on the Edge](https://arxiv.org/abs/2110.14032)


# CIFAR-10 and CIFAR-100

## Requirements

python >= 3.6

PyTorch >= 1.6

TorchVision >= 0.7

Other required dependency: `numpy`, `pyyaml`, `pickle`.


## Dynamic Sparse Training with MEST
We provide two different training modes:

1. Training with Elastic Mutation (MEST+EM)
2. Training with Soft Memory Bound Elastic Mutation (MEST+EM&S)

- The sample training script can be found by:
    ```
    cd scripts/resnet32/EM
    cd scripts/resnet32/EM_S
    ```

Basically, we implement our MEST based on a 'prune-and-grow' mechanism. 
So the two modes can be controlled by setting different 'UPPER_BOUND' and 'LOWER_BOUND' sparsity as shown in the training scripts.

For example, the target sparsity is 90% and the mutation ratio gradually decreases from 2% to 1% and finally to 0%.:
- For MEST+EM:
    
    ```
    LOWER_BOUND="0.92-0.91-0.9"
    UPPER_BOUND="0.9-0.9-0.9"
    ```
- For MEST+EM&S:
    ```
    LOWER_BOUND="0.9-0.9-0.9"
    UPPER_BOUND="0.88-0.89-0.9"
    ```

A configure file is used to specify which layers are conducting sparse training.
For the layers that are not specified will remain dense.
- An example in the script:
    ```
    CONFIG_FILE=$"./profiles/resnet32_cifar/blk/resnet32_0.9.yaml"
    ```

#### Sparsity Type
We provide four different sparsity schemes that can be selected by setting 'SPARSITY_TYPE' variable in the script, including:

- unstructured sparsity: ```irregular```
- pattern-based sparsity: ```pattern+filter_balance```
- block-based sparsity: ```free_block_prune_column_4```
- channel sparsity: ```channel```

#### Mutation Configurations
The mutation frequency can be controlled by 'SP_MASK_UPDATE_FREQ' variable
and the time (epoch) to decrease the mutation ratio is set by 'MASK_UPDATE_DECAY_EPOCH'.

- E.g. mutate every 5 epochs and mutation ratio decreases at 90 and 120 epoch, respectively:
    ```
    MASK_UPDATE_DECAY_EPOCH="90-120"
    SP_MASK_UPDATE_FREQ="5"
    ```

#### Training with Data Efficiency
We investigate the data-efficient training in sparse training scene.
To evaluate the importance of the training data (sample), 
we follow the method in prior work 
[An Empirical Study of Example Forgetting during Deep Neural Network Learning](https://arxiv.org/abs/1812.05159)

We incorporate it into our MEST code and the example training script can be found in
```
cd scripts/resnet32/data_efficient
```

And a sorted training data importance result is stored in a ```.pkl``` file, which can be directly used in our data-efficient MEST training.
The ```.pkl``` is provided in ```./pkl/``` directory.


# Citation
if you find this repo is helpful, please cite
```
@article{yuan2021mest,
  title={MEST: Accurate and Fast Memory-Economic Sparse Training Framework on the Edge},
  author={Yuan, Geng and Ma, Xiaolong and Niu, Wei and Li, Zhengang and Kong, Zhenglun and Liu, Ning and Gong, Yifan and Zhan, Zheng and He, Chaoyang and Jin, Qing and others},
  journal={Advances in Neural Information Processing Systems},
  volume={34},
  year={2021}
}

```

