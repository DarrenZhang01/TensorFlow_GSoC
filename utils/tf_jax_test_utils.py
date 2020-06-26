# Copyright 2018 Google LLC
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     https://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

"""
Reconstruct the JAX test utils
  {_default_tolerance, device_under_test, JaxTestCase, cases_from_list}
such that they are under TF support.

Adapted based on <https://github.com/google/jax/blob/master/jax/test_util.py>,

Zhibo Zhang, 2020.06.26
"""
