## Neural Tangents (Infinite-width NNs) for TensorFlow 2.x.

[![Build Status](https://travis-ci.com/DarrenZhang01/Neural_Tangents_TensorFlow.svg?branch=master)](https://travis-ci.com/DarrenZhang01/Neural_Tangents_TensorFlow)

TensorFlow GSoC program '20 - Zhibo Zhang

Mentors: Ashish Agarwal, Allen Lavoie, Dan Moldovan, Peng Wang, Paige Bailey and Akshay Naresh Modi @ Google Brain

Also special thanks to the Google Brain Researchers Roman Novak and Sam Schoenholz for their assistance.

Neural Tangents (Infinite-width NNs) migration and reconstruction for TensorFlow 2.x, originally based on JAX (https://github.com/google/neural-tangents). The basic idea is when the width of the NNs approaches infinity, the dynamics is very similar to a Gaussian Process, which enables better understanding of Deep Learning. We hope with the help of enriched TF ecosystems, this can potentially power more SOTA research in explainable AI and assist in building trustworthy machine learning systems.

We welcome any thoughts and ideas - zhibozhang@cs.toronto.edu

<strong>UPDATE</strong>: Until now, the major APIs of Neural Tangents program are under TF support. Feel free to
run the [example files `function_space`, `infinite_fcn` and `weight_space`](https://github.com/DarrenZhang01/TensorFlow_GSoC/tree/master/neural-tangents/examples)


### 1. Pull Requests:

- https://github.com/google/trax/pull/956
- https://github.com/tensorflow/tensorflow/pull/42539
- https://github.com/google/neural-tangents/pull/59
- https://github.com/google/neural-tangents/pull/39

### 2. Commits:
- https://github.com/google/trax/pull/956/commits/a82fbcabecac01c195104780e16888c12bc2ffd9
- https://github.com/google/trax/pull/956/commits/08ad9f0a06e84c175dd9a5cb3902296eb84f3ff4
- https://github.com/google/trax/pull/956/commits/d31355a8bbd3167cc3f98f616f706af364c046ac
- https://github.com/google/trax/pull/956/commits/f41b2c81eb6882e8655688b444998c7c253d2a9a
- https://github.com/google/trax/pull/956/commits/9dff2f7204d5f4e8921131bd95e5ff6cff17672e
- https://github.com/google/trax/pull/956/commits/2e2c614f4876180f0e59ce04c39747f458665a1a
- https://github.com/google/trax/pull/956/commits/8a07811561d6c82c24eb6ddb042e8938221071e4
- https://github.com/google/trax/pull/956/commits/fef84d5ce0bf3f5ba89acb6cd4402c76fb323ff7
- https://github.com/google/trax/pull/956/commits/c880f86c39ac1b8372d610cba8118e7eb0fc0cc3


### 3. Other Contributions:
- Collected a list of JAX symbols that are used in Neural Tangents: https://github.com/google/neural-tangents/pull/39, where the according equivalent TF symbols are also listed;
- Collected a list of TF NumPy APIs - https://github.com/DarrenZhang01/Neural_Tangents_TensorFlow/blob/master/TF_Numpy_API.md, the list is still expanding.



### Reference:

Novak, R., Xiao, L., Hron, J., Lee, J., Alemi, A. A., Sohl-Dickstein, J., & Schoenholz, S. S. (2019). Neural tangents: Fast and easy infinite neural networks in python. arXiv preprint arXiv:1912.02803.
