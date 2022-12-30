import math
import os
import pandas as pd
import torch

from models import model_dict
import logging

A = "checkpoint/cifar10/24-May_resnet18_focal_loss_gamma=3.0" + "/model_best.pth"
B = "checkpoint/cifar10/23-May_resnet18_focal_loss_gamma=1.0" + "/model_best.pth"

csv_path = "results/student_same_metrics.csv"
model_name = "resnet18"
dataset = "cifar10"
teacher_name = ""
num_classes = 10

logging.basicConfig(level=logging.INFO, 
                        format="%(levelname)s:  %(message)s",
                        handlers=[
                            logging.StreamHandler()
                        ])

def fix_path(model_path):
    model_path = str(model_path)
    if os.path.isfile(model_path):
        return model_path
    checkpoint_splits = model_path.split("/")
    checkpoint_splits[0] = "mount"
    checkpoint_modified = "/".join(checkpoint_splits)
    return checkpoint_modified

def diff_l2(teacher, student):
    total = 0.
    for p1, p2 in zip(teacher.parameters(), student.parameters()):
        teacher_tensor = p1.data
        student_tensor = p2.data
        diff = torch.sum((teacher_tensor - student_tensor)**2)
        total += diff.item()
    return math.sqrt(total)

data_frame = pd.read_csv(csv_path)

# print(data_frame)
teachers = list()
students = list()

for i in range(len(data_frame)):
    if data_frame["dataset"][i] == dataset:
        teacher_path = fix_path(os.path.join("checkpoint", data_frame["dataset"][i], data_frame["teacher"][i], "model_best.pth"))
        # print(teacher_path, A, teacher_path == A)
        if teacher_path == B:
            # print("hello")
            if data_frame["temp"][i] == 1.5 and data_frame["dw"][i] == 0.05:

                logging.info(f"loading teacher model from: {teacher_path}")
                
                teacher = model_dict[model_name](num_classes=10)
                # load teacher model
                saved_model_dict = torch.load(teacher_path)
                assert saved_model_dict["dataset"] == dataset, \
                    "Teacher not trained with same dataset as the student"
                teacher.load_state_dict(saved_model_dict['state_dict'])
                
                student_path = os.path.join(data_frame["folder_path"][i], "model_best.pth")
                logging.info(f"loading student model from: {student_path}")
                student = model_dict[model_name](num_classes=10)
                # load teacher model
                saved_model_dict = torch.load(student_path)
                assert saved_model_dict["dataset"] == dataset, \
                    "student not trained with same dataset as the teacher"
                student.load_state_dict(saved_model_dict['state_dict'])

                print("hyper-params are:", data_frame["temp"][i], data_frame["dw"][i])
                print(f"l2 diff is : {diff_l2(teacher, student):.4f}")
                print()

                

# for x in teachers:
#     for y in students:
#         print(diff_l2(x, y))
