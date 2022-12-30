# KD Calibration
## Experiment - 1:   
Confirming if KD distilled models are "better" (in terms of calibration & accuracy) than the models that are produced with train-time calibration techniques. I am comparing KD distilled models only with the train-time techniques, as according to their papers proposing those train-time techniques, they are supposed to perform best calibration, while maintaining a solid accuracy (loosely speaking).

### Changed Files:
I have made some small changes in train_teacher.py file and renamed it as train_scratch.py (Training a model from scratch, i.e., without distillation). Original file has been moved to ' Trash/ ' directory

### New Files:  
The following files have been added to prepare a "test space" for this experiment:

- **scratch_script.sh** : Run this file to first train all your models (currently supporting ResNet8, ResNet32 and ResNet56), with all losses: cross_entropy, NLL+MDCA, FL+MDCA and focal_loss, with different combinations of betas and gammas.  
All traned models are stored in trained_model_library/  
Example:
To run resnet8, 32 and 56 models with all losses, on cifar10 dataset, run the following script.
```
> bash scratch_script.sh --model resnet8@resnet32@resnet56 --dataset cifar10 --epochs 125
```

- **KD_script.sh** : Run this file after you have trained all your models (stored in trained_model_library/ directory), and you intend to use them as teachers for training your new models. In the arguments you can specify what models you want to use as students (in --student_model argument), and over them, what models you want to use as teachers (in --teacher_model).  
The output distilled models are saved in ' distilled_model_directory/ '.  
NOTE: YOU MIGHT WANT TO CHANGE ' CUDA_VISIBLE_DEVICES=0 ' FOR SOME COMMANDS INSIDE THE SCRIPT.  
Example:
```
> bash KD_script.sh --student_model resnet8@resnet32 --teacher_model resnet32_cross_entropy@resnet32_NLL+MDCA_beta=5.0@resnet32_FL+MDCA_gamma=2.0_beta=10.0@resnet32_focal_loss_gamma=2.0 --dataset cifar10 --epochs 125
```
NOTE (for above command): PLEASE STICK TO THIS SYNTAX ONLY FOR --student_model and --teacher_model VALUES.

- **find_checkpoint.sh** : You are not expected to worry about this file too much. Just get this - This script takes a description of a teacher model you want to use from your ' trained_library_checkpoint/ ' directory and automatically gets the **recently trained model** from there by reading out the proper model directory names in there, and outputs the exact checkpoint you want to use.  
THIS SCRIPT WAS PRIMARILY CREATED FOR KD_script.sh  
Example: (first argument is the teacher model's description. Second argument is the dataset, over which your teacher has been trained on.)
```
> bash find_checkpoint.sh resnet56_FL+MDCA_gamma=2.0_beta=10.0 cifar10
```
NOTE:  
1) FOR MORE INFORMATION ON THE SCRIPTS, CHECK OUT THE FILES, HOPEFULLY THEY ARE WELL COMMENTED.  
2) IN CASE OF ANY PROBLEMS, RAISE AN ISSUE, OR LET ME KNOW AT: kartik.anand.19031@iitgoa.ac.in
