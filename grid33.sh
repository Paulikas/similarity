#!/bin/bash

rm model_weights.h5
python kfold.py
python kfold-17.py
python top_model.py -r -e 200 -l 8
python top_model.py -s -l 8 >> res22

rm model_weights.h5
python kfold.py
python kfold-18.py
python top_model.py -r -e 200 -l 8
python top_model.py -s -l 8 >> res22

rm model_weights.h5
python kfold.py
python kfold-19.py
python top_model.py -r -e 200 -l 8
python top_model.py -s -l 8 >> res22