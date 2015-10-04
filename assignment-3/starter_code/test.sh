#!/bin/bash

# verbose
set -x


infile="models/RNN_wvecDim_25_step_1e-2_2.bin" # the pickled neural network
model="RNN" # the neural network type

echo $infile

# test the model on test data
#python runNNet.py --inFile $infile --test --data "test" --model $model

# test the model on dev data
python runNNet.py --inFile $infile --test --data "dev" --model $model

# test the model on training data
#python runNNet.py --inFile $infile --test --data "train" --model $model












