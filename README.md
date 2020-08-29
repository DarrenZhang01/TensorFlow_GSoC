# Neural Tangents (Infinite-width NNs) for TensorFlow 2.x.

[![Build Status](https://travis-ci.com/DarrenZhang01/TensorFlow_GSoC.svg?branch=master)](https://travis-ci.com/DarrenZhang01/TensorFlow_GSoC)

Mentors: [Ashish Agarwal](https://www.linkedin.com/in/ashish-agarwal-3932b764/), [Allen Lavoie](https://github.com/allenlavoie), [Peng Wang](https://github.com/wangpengmit), [Dan Moldovan](https://research.google/people/DanMoldovan/), [Paige Bailey](https://github.com/dynamicwebpaige) and [Akshay Naresh Modi](https://github.com/akshaym) @ Google Brain

Also special thanks to the Google Brain Researchers [Roman Novak](https://github.com/romanngg) and [Sam Schoenholz](https://github.com/sschoenholz) for their insights and assistance.

## Summary: TensorFlow Google Summer of Code program '20 - Zhibo Zhang

* <b>Motivation</b>: Neural Tangents (Infinite-width NNs) migration and reconstruction for TensorFlow 2.x, originally based on JAX (https://github.com/google/neural-tangents). The basic idea is when the width of the NNs approaches infinity, the dynamics is very similar to a Gaussian Process, which enables better understanding of Deep Learning. We hope with the help of enriched TF ecosystems, this can potentially power more SOTA research in explainable AI and assist in building trustworthy machine learning systems.

* <b>Contributions</b>: This is not a pure engineering project. Instead, this is an R\&D project. Neural Tangents itself is a work in progress. Besides, nobody ever tried any migration from JAX to TensorFlow before this project, which also increases the difficulty. Every research project has a chance of failure, but fortunately, after overcoming numerous difficulties, my mentors and I have finished migrating the major APIs from JAX to TensorFlow ([Pull Request 1](https://github.com/google/neural-tangents/pull/61) and [Pull Request 2](https://github.com/google/neural-tangents/pull/59)). <b>However, the meaning of this project is far more than a pure migration of Neural Tangents - it is about enriching the TensorFlow NumPy extensions ecosystem ([Pull Request 3](https://github.com/google/trax/pull/956) and [Pull Request 4](https://github.com/google/trax/pull/954)), about checking the usability of latest nightly version of TensorFlow and about exploring various possibilities on the compatibility and design differences between JAX and TensorFlow ([list of issue logs](https://github.com/DarrenZhang01/TensorFlow_GSoC/issues?q=is%3Aissue+is%3Aclosed), and an example of TensorFlow improvement, motivated by this migration - https://github.com/google/trax/pull/970).</b>

* <b>Timeline</b>:  In May, my mentors and I had several discussions about the latest and SOTA papers. In June, I mainly focused on the construction of helper APIs, like `tf_jax_stax`, `tf_lax`, general convolution, general dot, reduce window;
Starting from the end of June and in early July, I started the official reconstruction and migration of NT from JAX to TF, including setting up the Travis CI and integrating the tests; Then as the hardest part, I started debugging along with the migration.

* <b>Acknowledgement</b>: I am really lucky to collaborate with the Google Brain TensorFlow team, and get to be guided by my excellent mentors at TensorFlow. They always provide swift response and a lot of patience in helping me become a better problem solver. In particular, I want to thank [Ashish Agarwal](https://www.linkedin.com/in/ashish-agarwal-3932b764/), [Allen Lavoie](https://github.com/allenlavoie), [Peng Wang](https://github.com/wangpengmit), [Dan Moldovan](https://research.google/people/DanMoldovan/), [Paige Bailey](https://github.com/dynamicwebpaige) and [Akshay Naresh Modi](https://github.com/akshaym) for their guidance. In this project, I also get to collaborate and learn from the Google Brain Researchers [Roman Novak](https://github.com/romanngg) and [Sam Schoenholz](https://github.com/sschoenholz), and they assisted me in submitting the changes to the Google Neural Tangents repo. Thank you all for an execellent summer!

* <b>Future Work</b>: Although the major migration has been done, some tests about some utility files need to be run through and some potential compatibility issues need to be dealt with. I also plan to submit another Pull Request about TF `reduce_window` for TensorFlow NumPy extensions. Besides, I will improve the TF NumPy `size` API according to the code review in the Pull Request.

We welcome any thoughts and ideas - zhibozhang@cs.toronto.edu

<strong>UPDATE</strong>: Until now, the major APIs of Neural Tangents program are under TF support. Feel free to
run the [example files `function_space`, `infinite_fcn` and `weight_space`](https://github.com/DarrenZhang01/TensorFlow_GSoC/tree/master/neural-tangents/examples)


## Reference:

Novak, R., Xiao, L., Hron, J., Lee, J., Alemi, A. A., Sohl-Dickstein, J., & Schoenholz, S. S. (2019). Neural tangents: Fast and easy infinite neural networks in python. arXiv preprint arXiv:1912.02803.

## Appendix: 

### 1. Pull Requests:

- https://github.com/google/trax/pull/956
- https://github.com/google/trax/pull/954
- https://github.com/google/neural-tangents/pull/61
- https://github.com/google/neural-tangents/pull/59
- https://github.com/google/neural-tangents/pull/39
- https://github.com/google/neural-tangents/pull/62
- https://github.com/tensorflow/tensorflow/pull/42539

### 2. Commits:
- https://github.com/google/trax/commit/a82fbcabecac01c195104780e16888c12bc2ffd9
- https://github.com/google/trax/commit/08ad9f0a06e84c175dd9a5cb3902296eb84f3ff4
- https://github.com/google/trax/commit/d31355a8bbd3167cc3f98f616f706af364c046ac
- https://github.com/google/trax/commit/f41b2c81eb6882e8655688b444998c7c253d2a9a
- https://github.com/google/trax/commit/9dff2f7204d5f4e8921131bd95e5ff6cff17672e
- https://github.com/google/trax/commit/2e2c614f4876180f0e59ce04c39747f458665a1a
- https://github.com/google/trax/commit/8a07811561d6c82c24eb6ddb042e8938221071e4
- https://github.com/google/trax/commit/fef84d5ce0bf3f5ba89acb6cd4402c76fb323ff7
- https://github.com/google/trax/commit/c880f86c39ac1b8372d610cba8118e7eb0fc0cc3
- https://github.com/google/trax/commit/083ba340a4c513a7ac58647d22341accc07323d1
- https://github.com/google/trax/commit/acd76cc1e57bbbd36fad55cd134e8acc4d6e2637
- https://github.com/google/trax/commit/8c5a7b948d54e9d61d7dd7ffc033f60b9bd56560
- https://github.com/google/trax/commit/e169591adb24d506b26bfe94e1c02a959bf1e49d
- https://github.com/google/trax/commit/ea29343f9af70d7d20f0a1ab3cf56e40d993a074
- https://github.com/google/trax/commit/a13a238041d5fa760b13c9c0c72f9c9db9c0eeb5
- https://github.com/google/trax/commit/5887a7ac29a8420139bc2f09bc5e38848b9b4821
- https://github.com/google/trax/commit/9943cf0c38b2cb91df74940bd0afcaaad9a8bef2
- Commits about TF Neural Tangents: https://github.com/google/neural-tangents/commits/neural-tangents-tf


### 3. Other Contributions:
- Collected a list of JAX symbols that are used in Neural Tangents: https://github.com/DarrenZhang01/TensorFlow_GSoC/blob/master/SYMBOL_LIST.md, where the according equivalent TF symbols are also listed;
- Collected a list of TF NumPy APIs - https://github.com/DarrenZhang01/Neural_Tangents_TensorFlow/blob/master/TF_Numpy_API.md, the list is still expanding.



