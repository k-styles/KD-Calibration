import math
import os
import pandas as pd
import torch
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

sns.set_style("darkgrid")

from models import model_dict
import logging

A = "checkpoint/cifar10/24-May_resnet18_focal_loss_gamma=3.0" + "/model_best.pth"
B = "checkpoint/cifar10/23-May_resnet18_focal_loss_gamma=1.0" + "/model_best.pth"
C = "checkpoint/cifar10/23-May_resnet18_cross_entropy/model_best.pth"

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

df = pd.read_csv(csv_path)
# import pdb; pdb.set_trace()

# print(df)
teachers_path = {"A":A, "B":B, "C":C}

# considering A's students

current_student_to_consider = C
current_students_label = "C"
# current_student_to_consider = A
# current_students_label = "A"

for teacher_name in teachers_path:
    teacher_path = teachers_path[teacher_name]

    # load teacher
    teacher = model_dict[model_name](num_classes=10)
    saved_model_dict = torch.load(teacher_path)
    assert saved_model_dict["dataset"] == dataset, \
        "Teacher not trained with same dataset as the student"
    teacher.load_state_dict(saved_model_dict['state_dict'])

    df_t = df[df["teacher"] == current_student_to_consider.split("/")[2]]
    temp = df_t["temp"].unique()        
    dw = df_t["dw"].unique()

    # print(temp)
    # print(dw)
    # exit()

    temp, dw = np.meshgrid(temp, dw) 
    # import pdb; pdb.set_trace()   
    l, w = temp.shape
    norms = np.zeros((l,w))
    assert norms.shape == temp.shape

    # initialising the plot 
    # fig = plt.figure()
    # ax = plt.axes(projection='3d')

    # iterating over students
    for i in range(l):
        for j in range(w):
            student_path = df_t[(df_t["temp"] == temp[i][j]) & (df_t["dw"] == dw[i][j])]["folder_path"].unique()
            if len(student_path) == 0:
                continue
            assert len(student_path) == 1
            student_path = student_path[0]


            # load student
            student_path = os.path.join(student_path, "model_best.pth")
            logging.info(f"loading student model from: {student_path}")
            student = model_dict[model_name](num_classes=10)
            # load teacher model
            saved_model_dict = torch.load(student_path)
            assert saved_model_dict["dataset"] == dataset, \
                "student not trained with same dataset as the teacher"
            student.load_state_dict(saved_model_dict['state_dict'])

            norms[i][j] = diff_l2(teacher, student)

    plt.plot([i for i in range(l*w)], norms.reshape(-1), label = f"teacher = {teacher_name}")
plt.xlabel("Hyperparameter index")
plt.ylabel("L2 Norm")
plt.title(f"Comparision of {current_students_label}'s students")
# ax.plot_surface(temp, dw, norms)
# ax.plot_surface(temp, dw, norms, color = "red", shade = False)
plt.legend()
plt.savefig(f"graphs/l2_norm/{current_students_label = }_{model_name = }_{dataset = }.png")













# for i in range(len(df)):
#     if df["dataset"][i] == dataset:
#         teacher_path = fix_path(os.path.join("checkpoint", df["dataset"][i], df["teacher"][i], "model_best.pth"))
#         # print(teacher_path, A, teacher_path == A)
#         if teacher_path == B:
#             # print("hello")
#             if df["temp"][i] == 1.5 and df["dw"][i] == 0.05:

#                 logging.info(f"loading teacher model from: {teacher_path}")
                
#                 teacher = model_dict[model_name](num_classes=10)
#                 # load teacher model
#                 saved_model_dict = torch.load(teacher_path)
#                 assert saved_model_dict["dataset"] == dataset, \
#                     "Teacher not trained with same dataset as the student"
#                 teacher.load_state_dict(saved_model_dict['state_dict'])
                
#                 student_path = os.path.join(df["folder_path"][i], "model_best.pth")
#                 logging.info(f"loading student model from: {student_path}")
#                 student = model_dict[model_name](num_classes=10)
#                 # load teacher model
#                 saved_model_dict = torch.load(student_path)
#                 assert saved_model_dict["dataset"] == dataset, \
#                     "student not trained with same dataset as the teacher"
#                 student.load_state_dict(saved_model_dict['state_dict'])

#                 print("hyper-params are:", df["temp"][i], df["dw"][i])
#                 print(f"l2 diff is : {diff_l2(teacher, student):.4f}")
#                 print()

                

# # for x in teachers:
# #     for y in students:
# #         print(diff_l2(x, y))
