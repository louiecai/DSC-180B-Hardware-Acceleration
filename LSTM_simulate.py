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

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    
model = model.LSTMClassifier(52, 250, 25, 2).to(device)
model.load_state_dict(torch.load("./models/LSTM.pth")) 
with open ('small_sample', 'rb') as fp:
    data_feature = pickle.load(fp)
fp.close()
model.eval()

with profile(activities=[ProfilerActivity.CPU, ProfilerActivity.CUDA], profile_memory=True, record_shapes=True, with_stack=True) as prof:
    with record_function("model_inference"):
        #################### Profiling this part##########################
        

        input = torch.Tensor([data_feature[0]]).to(device)
        output = model(input)

        pred_index = torch.max(output, 1)[1]

        pred = torch.max(output, 1)[1]
        print("Activity ID: ", pred.item())
        ##################################################################

txt = prof.key_averages().table(sort_by="self_cpu_memory_usage")
path = "./profiling/LSTM/table_"+ configs.environment_name +".txt"
text_file = open(path, "w")
text_file.write(txt)
text_file.close()

path = "./profiling/LSTM/chromeTrace_"+ configs.environment_name +".json"
prof.export_chrome_trace(path)

path = "./profiling/LSTM/profiler_stacks_cuda_"+ configs.environment_name +".txt"
prof.export_stacks(path, "self_cuda_time_total")

path = "./profiling/LSTM/profiler_stacks_cpu_"+ configs.environment_name +".txt"
prof.export_stacks(path, "self_cpu_time_total")
