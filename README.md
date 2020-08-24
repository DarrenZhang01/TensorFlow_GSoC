## Neural Tangents (Infinite-width NNs) for TensorFlow 2.x.

[![Build Status](https://travis-ci.com/DarrenZhang01/Neural_Tangents_TensorFlow.svg?branch=master)](https://travis-ci.com/DarrenZhang01/Neural_Tangents_TensorFlow)

TensorFlow GSoC program '20 - Zhibo Zhang

Mentors: Ashish Agarwal, Allen Lavoie, Dan Moldovan and Peng Wang @ Google Brain

Also special thanks to the Google Brain Researchers Roman Novak and Sam Schoenholz for their assistance.

Neural Tangents (Infinite-width NNs) migration and reconstruction for TensorFlow 2.x, originally based on JAX (https://github.com/google/neural-tangents). The basic idea is when the width of the NNs approaches infinity, the dynamics is very similar to a Gaussian Process, which enables better understanding of Deep Learning. We hope with the help of enriched TF ecosystems, this can potentially power more SOTA research in explainable AI and assist in building trustworthy machine learning systems.

We welcome any thoughts and ideas - zhibozhang@cs.toronto.edu

<strong>UPDATE</strong>: Until now, the major APIs of Neural Tangents program are under TF support. Feel free to
run the [example files `function_space`, `infinite_fcn` and `weight_space`](https://github.com/DarrenZhang01/TensorFlow_GSoC/tree/master/neural-tangents/examples)

### Reference:

Novak, R., Xiao, L., Hron, J., Lee, J., Alemi, A. A., Sohl-Dickstein, J., & Schoenholz, S. S. (2019). Neural tangents: Fast and easy infinite neural networks in python. arXiv preprint arXiv:1912.02803.
