import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torchvision import datasets, transforms
import heapq
from intro_pytorch import *
import io
import contextlib
import re

def main():
    train_loader = get_data_loader()
    test_loader = get_data_loader(False)
    train_loader_type = "<class 'torch.utils.data.dataloader.DataLoader'>"
    if(str(type(train_loader))!=train_loader_type):
        print("Failed get_data_loader()")

    train_data='''Dataset FashionMNIST
    Number of datapoints: 60000
    Root location: ./data
    Split: Train
    StandardTransform
Transform: Compose(
               ToTensor()
               Normalize(mean=(0.1307,), std=(0.3081,))
           )'''


    if(str(train_loader.dataset)!=train_data):
        print("Failed get_data_loader()")
    model = build_model()
    model_string = '''Sequential(
  (0): Flatten(start_dim=1, end_dim=-1)
  (1): Linear(in_features=784, out_features=128, bias=True)
  (2): ReLU()
  (3): Linear(in_features=128, out_features=64, bias=True)
  (4): ReLU()
  (5): Linear(in_features=64, out_features=10, bias=True)
)'''
    if(str(model)!=model_string):
        print("Failed build_model()")

    criterion = nn.CrossEntropyLoss()
    f = io.StringIO()
    with contextlib.redirect_stdout(f):
        train_model(model,train_loader,criterion,1)
    output = f.getvalue()
    epoch_regex = r"Train Epoch:\s+0\s+Accuracy:\s+\d{5}\/60000\(\d{2}\.\d{2}%\)\s+Loss:\s+\d\.\d{3}"
    if(len(re.findall(epoch_regex,output))==0):
        print("Failed train_model()")

    f = io.StringIO()
    with contextlib.redirect_stdout(f):
        evaluate_model(model,test_loader,criterion,True)
    output = f.getvalue()

    eval_regex = r"Average loss:\s+\d\.\d{4}\nAccuracy:\s+\d{2}\.\d{2}%"
    if(len(re.findall(eval_regex,output))==0):
        print("Failed evaluate_model()")

    eval_regex = r"Accuracy:\s+\d{2}\.\d{2}%"
    f = io.StringIO()
    with contextlib.redirect_stdout(f):
        evaluate_model(model,test_loader,criterion,False)
    output = f.getvalue()
    if(len(re.findall(eval_regex,output))==0):
        print("Failed evaluate_model()")

    pred_set, _ = next(iter(get_data_loader(False)))
    f = io.StringIO()
    with contextlib.redirect_stdout(f):
        predict_label(model, pred_set, 1)
    output = f.getvalue()
 
    regex_label = r"Pullover: \d{2}\.\d{1,2}%\nShirt: \d{1,2}\.\d{2}%\nCoat: \d{1,2}\.\d{2}%"
    if(len(re.findall(regex_label,output))==0):
        print("Failed predict_label()")



    

    

main()

