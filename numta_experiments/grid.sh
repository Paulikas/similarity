#!/bin/bash

for i in {1..100}
do
  rm model_weights.h5
  python top_model.py -r -e 200 -l 8 > /dev/null
  python top_model.py -s -l 8 >> res_ft
done

