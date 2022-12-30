# streamlit run streamlit_chain_exp1.py

import streamlit as st
import pandas as pd
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import os

sns.set_theme()
plt.rcParams["figure.figsize"] = (12,6)

st.title('Chaining Exp graphs (dw = 0.99)')


@st.cache
def load_data(nrows):
    pass

def get_plot(df, student_folder):
    print(f"{student_folder = }")
    newdf = df[df["folder_path"] == student_folder]
    teacher_substring = newdf["teacher"].unique()[0]
    print(teacher_substring)
    if "cifar100" in student_folder:
        if teacher_substring in teachers["cifar100"]["resnet18"]["Uncalibrated Teacher"]:
            teacher_folder_path = os.path.dirname(teachers["cifar100"]["resnet18"]["Uncalibrated Teacher"])
        elif teacher_substring in teachers["cifar100"]["resnet18"]["Calibrated Teacher"]:
            teacher_folder_path = os.path.dirname(teachers["cifar100"]["resnet18"]["Calibrated Teacher"])
    elif "cifar10" in student_folder:
        if teacher_substring in teachers["cifar10"]["resnet18"]["Uncalibrated Teacher"]:
            teacher_folder_path = os.path.dirname(teachers["cifar10"]["resnet18"]["Uncalibrated Teacher"])
        elif teacher_substring in teachers["cifar10"]["resnet18"]["Calibrated Teacher"]:
            teacher_folder_path = os.path.dirname(teachers["cifar10"]["resnet18"]["Calibrated Teacher"])

    teacher_df = pd.read_table(os.path.join(teacher_folder_path,"train_metrics.txt"))
    student_df = pd.read_table(os.path.join(student_folder,"train_metrics.txt"))

    teacher_df[teacher_df["lr"] != "best_accuracy"]
    teacher_df[teacher_df["lr"] != "best_calibration"]
    student_df[student_df["lr"] != "best_accuracy"]
    student_df[student_df["lr"] != "best_calibration"]

    plt.clf()
    # metric = "test_acc"
    # # metric = "SCE"
    # # metric = "ECE"
    student_data_to_plot = student_df[metric].tolist()
    teacher_data_to_plot_basic = teacher_df[metric].tolist()
    teacher_data_to_plot = teacher_data_to_plot_basic.copy()
    immediate_teacher_data_to_plot = teacher_data_to_plot_basic.copy()
    epoch_len = len(teacher_data_to_plot)

    # since teachers data is only a fraction of students data we make more for visual purposes
    assert len(student_data_to_plot) % len(teacher_data_to_plot) == 0
    factor  = len(student_data_to_plot) // len(teacher_data_to_plot)

    for i in range(1, factor):
        teacher_data_to_plot += teacher_data_to_plot_basic
        immediate_teacher_data_to_plot += student_data_to_plot[(i-1)*epoch_len: i*epoch_len]
        plt.axvline(x = i*epoch_len, linestyle = "dashdot", color = "black", zorder = 100, linewidth=2) 

    assert len(student_data_to_plot) == len(teacher_data_to_plot) 

    if(show_student):
        plt.plot(list(range(len(student_data_to_plot))), student_data_to_plot , label = "chained students", zorder = 5,)
        if(metric == "test_acc"):
            st.write(f"{max(student_data_to_plot) = }")
        else:
            st.write(f"{min(student_data_to_plot) = }")

    if(show_og_teacher):
        plt.plot(list(range(len(teacher_data_to_plot))), teacher_data_to_plot , label = "og teacher")
        if(metric == "test_acc"):
            st.write(f"{max(teacher_data_to_plot) = }")
        else:
            st.write(f"{min(teacher_data_to_plot) = }")
    if(show_immediate_prev_teacher):
        plt.plot(list(range(len(immediate_teacher_data_to_plot))), immediate_teacher_data_to_plot , label = "immediate prev teacher", color = "red")
        if(metric == "test_acc"):
            st.write(f"{max(immediate_teacher_data_to_plot) = }")
        else:
            st.write(f"{min(immediate_teacher_data_to_plot) = }")
    
    plt.title(f"Comparing {metric}")
    plt.legend()
    plt.tight_layout()

    return plt

teachers = {
    "cifar10": {
        "resnet18" : {
            "Uncalibrated Teacher": "checkpoint/cifar10/24-May_resnet18_focal_loss_gamma=3.0/model_best.pth", # unCalibrated
            "Calibrated Teacher": "checkpoint/cifar10/23-May_resnet18_focal_loss_gamma=1.0/model_best.pth", # calibrated
            # "checkpoint/cifar10/23-May_resnet18_cross_entropy/model_best.pth" # CE
        },
        # "resnet34" : [
        #     "checkpoint/cifar10/24-May_resnet34_focal_loss_gamma=3.0/model_best.pth", # least calibrated
        #     "checkpoint/cifar10/23-May_resnet34_FL+MDCA_gamma=2.0_beta=1.0/model_best.pth", # Calibrated
        # ]
    },
    "cifar100" : {
        "resnet18" : {
            "Uncalibrated Teacher": "checkpoint/cifar100/24-May_resnet18_cross_entropy/model_best.pth", # least calibrated
            "Calibrated Teacher": "checkpoint/cifar100/24-May_resnet18_focal_loss_gamma=1.0/model_best.pth" # Calibrated
        },
        # "resnet34" : [
        #     "checkpoint/cifar100/24-May_resnet34_cross_entropy/model_best.pth", # least calibrated
        #     "checkpoint/cifar100/24-May_resnet34_FL+MDCA_gamma=2.0_beta=5.0/model_best.pth" # Calibrated
        # ]
    }
}

students = ["resnet18"] #, "resnet34"]
# datasets = ["cifar10"]
datasets = ["cifar10", "cifar100"]

CHAIN_LENGTH = 5

# please confirm this @neelabh
temps = [1.0, 1.5, 2.0, 3.0, 4.5, 6.0, 8.0, 10.0, 20.0]
# dws = [0.99, 0.95, 0.5, 0.1, 0.05]
dws = [0.99]

metrics = ["test_acc", "SCE", "ECE"]

calibration_choices = ["Calibrated Teacher", "Uncalibrated Teacher"]


dataset = st.sidebar.radio("Dataset",datasets)
calibration_choice = st.sidebar.radio("Calibration Choice",calibration_choices)
temp = st.sidebar.radio("Temp",temps)
metric = st.sidebar.radio("Metric",metrics)

show_student = st.sidebar.checkbox("Students",  True)
show_og_teacher = st.sidebar.checkbox("OG Teacher", True)
show_immediate_prev_teacher = st.sidebar.checkbox("Immediate Prev Teacher", True)

student = "resnet18" # Hard coded

# st.write(calibration_choice)

file_name = "results/chained_student_metrics.csv"
df = pd.read_csv(file_name)

# import pdb; pdb.set_trace()

trained_students_list = df["folder_path"].unique().tolist()

for student_folder in trained_students_list:
    # st.write(student_folder)
    if (teachers[dataset][student][calibration_choice].split("/")[2]) in student_folder and (f"temp={temp}" in student_folder):
        st.write("Reading from folder:")
        st.write(student_folder)
        # st.write("okaoka")
        

        st.pyplot(get_plot(df, student_folder))
        # st.pyplot(get_plot_comparing_prev_teacher(df, student_folder))
        # plt.savefig(os.path.join(student_folder,f"chained_compare_{metric}.svg"))


