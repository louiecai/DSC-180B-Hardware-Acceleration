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


def cnn_simulate(env, dev, trial):
    device = torch.device("cpu")
    GPU = False
    if dev == "gpu":
        device = torch.device("cuda")
        GPU = True
    
        
    model = model_util.CNN(25, 300, 52).to(device)
    if GPU:
        model.load_state_dict(torch.load("./models/cnn_model.pth")) 
    else:
        model.load_state_dict(torch.load("./models/cnn_model.pth",map_location=torch.device('cpu')))   

    with open ('small_sample', 'rb') as fp:
        data_feature = pickle.load(fp)
    fp.close()
    model.eval()

    act = [ProfilerActivity.CPU]
    if GPU:
        act.append(ProfilerActivity.CUDA)

    with profile(activities=act, profile_memory=True, record_shapes=True, with_stack=True) as prof:
        with record_function("model_inference"):
            #################### Profiling this part##########################
            for i in range(10):
                input = torch.Tensor([data_feature[i]]).to(device)
                output = model(input)

                pred = torch.max(output, 1)[1]
                print("Activity ID: ", pred.item())
            ##################################################################

    environment_name = env
    rpath = "./profiling/"+ environment_name +"/trial_"+trial+"/cpu/CNN/"
    if GPU:
        rpath = "./profiling/"+ environment_name +"/trial_"+trial+"/gpu/CNN/"
    elif dev=="other":
        rpath = "./profiling/"+ environment_name +"/trial_"+trial+"/other/CNN/"

    txt = prof.key_averages().table(sort_by="cpu_time_total")
    path = rpath +"time.txt"
    text_file = open(path, "w")
    text_file.write(txt)
    text_file.close()

    txt = prof.key_averages(group_by_input_shape=True).table()
    path = rpath +"shape.txt"
    text_file = open(path, "w")
    text_file.write(txt)
    text_file.close()

    path = rpath +"chromeTrace.json"
    prof.export_chrome_trace(path)

    txt = prof.key_averages(group_by_stack_n=5).table()
    path = rpath +"test.txt"
    text_file = open(path, "w")
    text_file.write(txt)
    text_file.close()