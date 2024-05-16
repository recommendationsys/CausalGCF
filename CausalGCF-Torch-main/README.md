# CausalGCF-Torch
This is the PyTorch implementation for our SIGIR 2021 paper. 

>Jiancan Wu, Xiang Wang, Fuli Feng, Xiangnan He, Liang Chen, Jianxun Lian,and Xing Xie. 2021. Self-supervised Graph Learning for Recommendation, [Paper in arXiv](https://arxiv.org/abs/2010.10783).

This project is based on [NeuRec](https://github.com/wubinzzu/NeuRec/tree/v3.x). Thanks to the contributors.

## Environment Requirement

The code runs well under python 3.7.7. The required packages are as follows:

- pytorch == 1.9.1
- numpy == 1.20.3
- scipy == 1.7.1
- pandas == 1.3.4
- cython == 0.29.24

## Quick Start
**Firstly**, compline the evaluator of cpp implementation with the following command line:

```bash
python local_compile_setup.py build_ext --inplace
```

If the compilation is successful, the evaluator of cpp implementation will be called automatically.
Otherwise, the evaluator of python implementation will be called.

**Note that the cpp implementation is much faster than python.**

**Secondly**, change the value of variable *root_dir* and *data_dir* in *main.py*, then specify dataset and recommender in configuration file *NeuRec.ini*.

Model specific hyperparameters are in configuration file *./conf/CausalGCF.ini*.

