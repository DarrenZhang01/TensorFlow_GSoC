#!/bin/bash
for f in neural-tangents/neural_tangents/tests/*.py; do 
  rm "${f%.*}.txt"
  echo "python3.8 $f &> ${f%.*}.txt"
  python3.8 "$f" &> "${f%.*}.txt" &
done
