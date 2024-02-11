# number of consecutive data to feed to model, since sensor get datas at frequency 100Hz, 300 here means we get latest 3 seconds datas.
seq_length = 300 

# When we get n~n+300 sequences, the next step we get n+40~n+340 sequences. The goal of this method is to reduce train set size
seq_interval = 40 


# even if we set sequence interval, the train set size is still large, that sometimes causes cuda out of memory
# here, after we shuffle the data set, we get 50% of data set. 
training_size_scale = 0.5 

#Define environment we simulate on
environment_name = "env1"
enable_GPU = True

#parameter for CNN
cnn_epoch = 50
cnn_learning_rate = 0.0001

#parameter for DNN
dnn_epoch = 50
dnn_learning_rate = 0.0001
