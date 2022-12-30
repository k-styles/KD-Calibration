import math
import os
import pandas as pd
import torch
from torch.utils.tensorboard import SummaryWriter

from models import model_dict
import logging

T1_path = "checkpoint/cifar10/24-May_resnet18_focal_loss_gamma=3.0/model_best.pth"
S1_path = "checkpoint/cifar10/students/07-Jun_teacher=24-May_resnet18_focal_loss_gamma=3.0_student=resnet18_temp=20.0_dw=0.99_runid=85/model_best.pth"
T2_path = "checkpoint/cifar10/23-May_resnet18_focal_loss_gamma=1.0/model_best.pth"
S2_path = "checkpoint/cifar10/students/06-Jun_teacher=23-May_resnet18_focal_loss_gamma=1.0_student=resnet18_temp=1.5_dw=0.05_runid=9/model_best.pth"

csv_path = "results/student_same_metrics.csv"
model_name = "resnet18"
dataset = "cifar10"
teacher_name = ""
num_classes = 10


writer = SummaryWriter("runs/T1-S1-T2-S2")
model_paths = {"T1": T1_path,"S1": S1_path,"T2": T2_path,"S2": S2_path }
i = 0
for exp_model_name in model_paths:

    logs_path = os.path.join("runs", exp_model_name)
    model_path = model_paths[exp_model_name]
    

    
    # load model
    model = model_dict[model_name](num_classes=10)
    saved_model_dict = torch.load(model_path)
    assert saved_model_dict["dataset"] == dataset, \
        "Teacher not trained with same dataset as the student"
    model.load_state_dict(saved_model_dict['state_dict'])
    for name, param in model.named_parameters():
        print(name)
        writer.add_histogram(name, param, i)
    i+=1
