import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import os

sns.set_theme()
plt.rcParams["figure.figsize"] = (12,6)

teachers = {
    "cifar10": {
        "resnet18" : [
            "checkpoint/cifar10/24-May_resnet18_focal_loss_gamma=3.0/model_best.pth", # unCalibrated
            "checkpoint/cifar10/23-May_resnet18_focal_loss_gamma=1.0/model_best.pth", # calibrated
            # "checkpoint/cifar10/23-May_resnet18_cross_entropy/model_best.pth" # CE
        ],
        # "resnet34" : [
        #     "checkpoint/cifar10/24-May_resnet34_focal_loss_gamma=3.0/model_best.pth", # least calibrated
        #     "checkpoint/cifar10/23-May_resnet34_FL+MDCA_gamma=2.0_beta=1.0/model_best.pth", # Calibrated
        # ]
    },
    "cifar100" : {
        "resnet18" : [
            "checkpoint/cifar100/24-May_resnet18_cross_entropy/model_best.pth", # least calibrated
            "checkpoint/cifar100/24-May_resnet18_focal_loss_gamma=1.0/model_best.pth" # Calibrated
        ],
        # "resnet34" : [
        #     "checkpoint/cifar100/24-May_resnet34_cross_entropy/model_best.pth", # least calibrated
        #     "checkpoint/cifar100/24-May_resnet34_FL+MDCA_gamma=2.0_beta=5.0/model_best.pth" # Calibrated
        # ]
    }
}


# open file with pandas
file_name = "results/chained_student_metrics.csv"
df = pd.read_csv(file_name)

# import pdb; pdb.set_trace()

trained_students_list = df["folder_path"].unique().tolist()

for student_folder in trained_students_list:
    print(f"{student_folder = }")
    newdf = df[df["folder_path"] == student_folder]
    teacher_substring = newdf["teacher"].unique()[0]
    print(teacher_substring)
    if "cifar100" in student_folder:
        if teacher_substring in teachers["cifar100"]["resnet18"][0]:
            teacher_folder_path = os.path.dirname(teachers["cifar100"]["resnet18"][0])
        elif teacher_substring in teachers["cifar100"]["resnet18"][1]:
            teacher_folder_path = os.path.dirname(teachers["cifar100"]["resnet18"][1])
    elif "cifar10" in student_folder:
        if teacher_substring in teachers["cifar10"]["resnet18"][0]:
            teacher_folder_path = os.path.dirname(teachers["cifar10"]["resnet18"][0])
        elif teacher_substring in teachers["cifar10"]["resnet18"][1]:
            teacher_folder_path = os.path.dirname(teachers["cifar10"]["resnet18"][1])

    teacher_df = pd.read_table(os.path.join(teacher_folder_path,"train_metrics.txt"))
    student_df = pd.read_table(os.path.join(student_folder,"train_metrics.txt"))

    teacher_df[teacher_df["lr"] != "best_accuracy"]
    teacher_df[teacher_df["lr"] != "best_calibration"]
    student_df[student_df["lr"] != "best_accuracy"]
    student_df[student_df["lr"] != "best_calibration"]

    plt.clf()
    metric = "test_acc"
    # metric = "SCE"
    # metric = "ECE"
    student_data_to_plot = student_df[metric].tolist()
    teacher_data_to_plot_basic = teacher_df[metric].tolist()
    teacher_data_to_plot = teacher_data_to_plot_basic.copy()
    epoch_len = len(teacher_data_to_plot)

    # since teachers data is only a fraction of students data we make more for visual purposes
    assert len(student_data_to_plot) % len(teacher_data_to_plot) == 0
    factor  = len(student_data_to_plot) // len(teacher_data_to_plot)

    for i in range(1, factor):
        teacher_data_to_plot += teacher_data_to_plot_basic
        plt.axvline(x = i*epoch_len, linestyle = "-") 

    assert len(student_data_to_plot) == len(teacher_data_to_plot) 

    plt.plot(list(range(len(student_data_to_plot))), student_data_to_plot , label = "chained students")
    plt.plot(list(range(len(teacher_data_to_plot))), teacher_data_to_plot , label = "teacher")
    
    plt.title(f"Comparing {metric}")
    plt.legend()
    plt.tight_layout()
    plt.savefig(os.path.join(student_folder,f"chained_compare_{metric}.svg"))