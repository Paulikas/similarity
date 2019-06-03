#!/bin/bash


rm model_weights.h5
python kfold.py
python kfold-14.py
python top_model.py -r -e 50 -l 12
python top_model.py -s -l 12 >> res50

rm model_weights.h5
python kfold.py
python kfold-15.py
python top_model.py -r -e 50 -l 12
python top_model.py -s -l 12 >> res50

rm model_weights.h5
python kfold.py
python kfold-16.py
python top_model.py -r -e 50 -l 12
python top_model.py -s -l 12 >> res50

rm model_weights.h5
python kfold.py
python kfold-17.py
python top_model.py -r -e 50 -l 12
python top_model.py -s -l 12 >> res50

rm model_weights.h5
python kfold.py
python kfold-18.py
python top_model.py -r -e 50 -l 12
python top_model.py -s -l 12 >> res50

rm model_weights.h5
python kfold.py
python kfold-19.py
python top_model.py -r -e 50 -l 12
python top_model.py -s -l 12 >> r50