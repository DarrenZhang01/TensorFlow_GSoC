#!/bin/bash
#for f in neural-tangents/neural_tangents/tests/*.py; do 
#  rm "${f%.*}.txt"
#  echo "python3.8 $f &> ${f%.*}.txt"
#  python3.8 "$f" &> "${f%.*}.txt" &
#done
python3.8 tests/infinite_fcn_test.py &
python3.8 tests/weight_space_test.py &
python3.8 tests/function_space_test.py
