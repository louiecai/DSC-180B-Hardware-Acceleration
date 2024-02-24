import torch
import pickle
import configs
import argparse
import os
import sys
from LSTM_simulate import LSTM_simulate
from cnn_simulate import cnn_simulate
from dnn_simulate import dnn_simulate

def main(args):
    print("GPU availability: {}".format(torch.cuda.is_available()))
    GPU = False
    FPGA = False

    if args.gpu=="True" and torch.cuda.is_available():
        GPU = True
    elif args.gpu=="True":
        print("GPU is not available on this environment")
    if args.fpga=="True":
        FPGA = True
    env = args.env
    path = "profiling/"+env+"/"
    

    if FPGA:
        os.makedirs(path+"fpga/CNN")
        cnn_simulate(env,"fpga")
        os.makedirs(path+"fpga/DNN")
        dnn_simulate(env,"fpga")
        os.makedirs(path+"fpga/LSTM")
        LSTM_simulate(env,"fpga")
    else:
        os.makedirs(path+"cpu/CNN")
        cnn_simulate(env,"cpu")
        os.makedirs(path+"cpu/DNN")
        dnn_simulate(env,"cpu")
        os.makedirs(path+"cpu/LSTM")
        LSTM_simulate(env,"cpu")



    if GPU:
        os.makedirs(path+"gpu/CNN")
        cnn_simulate(env,"gpu")
        os.makedirs(path+"gpu/DNN")
        dnn_simulate(env,"gpu")
        os.makedirs(path+"gpu/LSTM")
        LSTM_simulate(env,"gpu")



if __name__ == "__main__":
    #Parse the input arguments
    parser = argparse.ArgumentParser()
    parser.add_argument('--env', type=str)
    parser.add_argument('--gpu', type=str)
    parser.add_argument('--fpga', type=str)
    args = parser.parse_args()
    main(args)