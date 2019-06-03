#!/bin/bash
rm model_weights.h5
python kfold.py
python kfold-1.py
python top_model.py -r -e 200 -l 8 > /dev/null
python top_model.py -s -l 8 >> res22

rm model_weights.h5
python kfold.py
python kfold-2.py
python top_model.py -r -e 200 -l 8 > /dev/null
python top_model.py -s -l 8 >> res22

rm model_weights.h5
python kfold.py
python kfold-3.py
python top_model.py -r -e 200 -l 8 > /dev/null
python top_model.py -s -l 8 >> res22

rm model_weights.h5
python kfold.py
python kfold-4.py
python top_model.py -r -e 200 -l 8 > /dev/null
python top_model.py -s -l 8 >> res22

rm model_weights.h5
python kfold.py
python kfold-5.py
python top_model.py -r -e 200 -l 8 > /dev/null
python top_model.py -s -l 8 >> res22

rm model_weights.h5
python kfold.py
python kfold-6.py
python top_model.py -r -e 200 -l 8 > /dev/null
python top_model.py -s -l 8 >> res22

rm model_weights.h5
python kfold.py
python kfold-7.py
python top_model.py -r -e 200 -l 8 > /dev/null
python top_model.py -s -l 8 >> res22

rm model_weights.h5
python kfold.py
python kfold-8.py
python top_model.py -r -e 200 -l 8 > /dev/null
python top_model.py -s -l 8 >> res22

rm model_weights.h5
python kfold.py
python kfold-9.py
python top_model.py -r -e 200 -l 8 > /dev/null
python top_model.py -s -l 8 >> res22

rm model_weights.h5
python kfold.py
python kfold-10.py
python top_model.py -r -e 200 -l 8 > /dev/null
python top_model.py -s -l 8 >> res22
