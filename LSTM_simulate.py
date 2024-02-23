import torch
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import random
import torchvision
from typing import Optional, Union
import pickle
import model
import configs
from torch.profiler import profile, record_function, ProfilerActivity
import argparse


def main(args):
    device = torch.device("cpu")
    if args.device == "gpu":
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print("GPU availabel: {}, Current device: {}".format(torch.cuda.is_available(), device))
        
    LSTM = model.LSTMClassifier(52, 250, 25, 2).to(device)
    if torch.cuda.is_available() and args.device == "gpu":
        LSTM.load_state_dict(torch.load("./models/LSTM.pth")) 
    else:
        LSTM.load_state_dict(torch.load("./models/LSTM.pth",map_location=torch.device('cpu'))) 

    with open ('small_sample', 'rb') as fp:
        data_feature = pickle.load(fp)
    fp.close()
    LSTM.eval()

    act = [ProfilerActivity.CPU]
    if args.device == "gpu" and torch.cuda.is_available():
        act.append(ProfilerActivity.CUDA)

    with profile(activities=act, profile_memory=True, record_shapes=True, with_stack=True) as prof:
        with record_function("model_inference"):
            #################### Profiling this part##########################
            

            input = torch.Tensor([data_feature[0]]).to(device)
            output = LSTM(input)

            pred_index = torch.max(output, 1)[1]

            pred = torch.max(output, 1)[1]
            print("Activity ID: ", pred.item())
            ##################################################################

    environment_name = args.env

    txt = prof.key_averages().table(sort_by="self_cpu_memory_usage")
    path = "./profiling/LSTM/table_"+ environment_name +".txt"
    text_file = open(path, "w")
    text_file.write(txt)
    text_file.close()

    path = "./profiling/LSTM/chromeTrace_"+ environment_name +".json"
    prof.export_chrome_trace(path)

    if args.device == "gpu" and torch.cuda.is_available():
        path = "./profiling/LSTM/profiler_stacks_cuda_"+ environment_name +".txt"
        prof.export_stacks(path, "self_cuda_time_total")

    path = "./profiling/LSTM/profiler_stacks_cpu_"+ environment_name +".txt"
    prof.export_stacks(path, "self_cpu_time_total")

if __name__ == "__main__":
    #Parse the input arguments
    print(1)
    parser = argparse.ArgumentParser()
    parser.add_argument('--env', type=str)
    parser.add_argument('--device', type=str)
    args = parser.parse_args()
    main(args)