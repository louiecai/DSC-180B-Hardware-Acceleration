import torch
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import random
import torchvision
from typing import Optional, Union
import pickle
import model_util
import configs
from torch.profiler import profile, record_function, ProfilerActivity
import argparse


def LSTM_simulate(env, dev):
    device = torch.device("cpu")
    GPU = False
    if dev == "gpu":
        device = torch.device("cuda")
        GPU = True
    
 
    LSTM = model_util.LSTMClassifier(52, 250, 25, 2).to(device)
    if GPU:
        LSTM.load_state_dict(torch.load("./models/LSTM.pth")) 
    else:
        LSTM.load_state_dict(torch.load("./models/LSTM.pth",map_location=torch.device('cpu'))) 

    with open ('small_sample', 'rb') as fp:
        data_feature = pickle.load(fp)
    fp.close()
    LSTM.eval()

    act = [ProfilerActivity.CPU]
    if GPU:
        act.append(ProfilerActivity.CUDA)

    with profile(activities=act, profile_memory=True, record_shapes=True, with_stack=True) as prof:
        with record_function("model_inference"):
            #################### Profiling this part##########################
            for i in range(10):
                input = torch.Tensor([data_feature[i]]).to(device)
                output = LSTM(input)

                pred = torch.max(output, 1)[1]
                print("Activity ID: ", pred.item())
            ##################################################################

    environment_name = env
    rpath = "./profiling/"+ environment_name +"/cpu/LSTM/"
    if GPU:
        rpath = "./profiling/"+ environment_name +"/gpu/LSTM/"
    elif dev=="fpga":
        rpath = "./profiling/"+ environment_name +"/fpga/LSTM/"

    txt = prof.key_averages().table(sort_by="self_cpu_memory_usage")
    path = rpath +"table.txt"
    text_file = open(path, "w")
    text_file.write(txt)
    text_file.close()

    path = rpath +"chromeTrace.json"
    prof.export_chrome_trace(path)

    if GPU:
        path = rpath +"profiler_stacks_cuda.txt"
        prof.export_stacks(path, "self_cuda_time_total")

    path = rpath +"profiler_stacks_cpu.txt"
    prof.export_stacks(path, "self_cpu_time_total")

