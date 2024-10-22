#!/bin/bash

# verbose
set -x
###################
# Update items below for each train/test
###################

# training params
epochs=${NEPOCH:-30}
step=1e-2
wvecDim=${WVECDIM:-30}

# for RNN2 only, otherwise doesnt matter
middleDim=${MIDDLEDIM:-30}

model="RNN2" #either RNN, RNN2, RNN3, RNTN, or DCNN


######################################################## 
# Probably a good idea to let items below here be
########################################################
if [ "$model" == "RNN2" ]; then
    outfile="models/${model}_wvecDim_${wvecDim}_middleDim_${middleDim}_step_${step}_2.bin"
else
    outfile="models/${model}_wvecDim_${wvecDim}_step_${step}_2.bin"
fi


echo $outfile


python -u runNNet.py --step $step --epochs $epochs --outFile $outfile \
                --middleDim $middleDim --outputDim 5 --wvecDim $wvecDim --model $model