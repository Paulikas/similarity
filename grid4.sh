#!/bin/bash
rm model_weights.h5
python kfold.py
python kfold-1.py
python top_model.py -r -e 400 -l 8
python top_model.py -s -l 8 >> res400pl

rm model_weights.h5
python kfold.py
python kfold-2.py
python top_model.py -r -e 400 -l 8
python top_model.py -s -l 8 >> res400pl

rm model_weights.h5
python kfold.py
python kfold-3.py
python top_model.py -r -e 400 -l 8
python top_model.py -s -l 8 >> res400pl

rm model_weights.h5
python kfold.py
python kfold-4.py
python top_model.py -r -e 400 -l 8
python top_model.py -s -l 8 >> res400pl

rm model_weights.h5
python kfold.py
python kfold-5.py
python top_model.py -r -e 400 -l 8
python top_model.py -s -l 8 >> res400pl

rm model_weights.h5
python kfold.py
python kfold-6.py
python top_model.py -r -e 400 -l 8
python top_model.py -s -l 8 >> res400pl

rm model_weights.h5
python kfold.py
python kfold-7.py
python top_model.py -r -e 400 -l 8
python top_model.py -s -l 8 >> res400pl

rm model_weights.h5
python kfold.py
python kfold-8.py
python top_model.py -r -e 400 -l 8
python top_model.py -s -l 8 >> res400pl

rm model_weights.h5
python kfold.py
python kfold-9.py
python top_model.py -r -e 400 -l 8
python top_model.py -s -l 8 >> res400pl

rm model_weights.h5
python kfold.py
python kfold-10.py
python top_model.py -r -e 400 -l 8
python top_model.py -s -l 8 >> res400pl

rm model_weights.h5
python kfold.py
python kfold-11.py
python top_model.py -r -e 400 -l 8
python top_model.py -s -l 8 >> res400pl

rm model_weights.h5
python kfold.py
python kfold-12.py
python top_model.py -r -e 400 -l 8
python top_model.py -s -l 8 >> res400pl

rm model_weights.h5
python kfold.py
python kfold-13.py
python top_model.py -r -e 400 -l 8
python top_model.py -s -l 8 >> res400pl

rm model_weights.h5
python kfold.py
python kfold-14.py
python top_model.py -r -e 400 -l 8
python top_model.py -s -l 8 >> res400

rm model_weights.h5
python kfold.py
python kfold-15.py
python top_model.py -r -e 400 -l 8
python top_model.py -s -l 8 >> res400

rm model_weights.h5
python kfold.py
python kfold-16.py
python top_model.py -r -e 800 -l 8
python top_model.py -s -l 8 >> res800

rm model_weights.h5
python kfold.py
python kfold-17.py
python top_model.py -r -e 800 -l 8
python top_model.py -s -l 8 >> res800

rm model_weights.h5
python kfold.py
python kfold-18.py
python top_model.py -r -e 800 -l 8
python top_model.py -s -l 8 >> res800

rm model_weights.h5
python kfold.py
python kfold-19.py
python top_model.py -r -e 800 -l 8
python top_model.py -s -l 8 >> res800