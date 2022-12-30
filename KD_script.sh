################ 							     		##############
################ PLEASE ONLY RUN THIS SCRIPT AFTER RUNNING scratch_script.sh COMPLETELY ##############
################							     		##############

########## TRAINING EACH MODEL TRAINED WITH NLL WITH EVERY POSSIBLE CALIBRATED/UNCALIBRATED TEACHER ##########
#
# EXAMPLE: (To run the script for multiple students and teachers, pass in a string like in this example for --student_model and --teacher_model arguments)
#
# NOTE: THE HYPERPARAMETER VALUES OF BETA AND GAMMA MUST ALWAYS BE FLOATING POINT
#
# > bash KD_script.sh --student_model resnet8@resnet32 --teacher_model resnet32_cross_entropy@resnet32_NLL+MDCA_beta=5.0@resnet32_FL+MDCA_gamma=2.0_beta=10.0@resnet32_focal_loss_gamma=2.0 --dataset cifar10 --epochs 1
#
#
############################################# ARGUMENT HANDLING #################################################
# Give the arguments in the following way:
# --teacher_model : {model_name_1}_{loss_1}_{loss_weights_1}@{model_name_2}_{loss_2}_{loss_weights_2} 
# Examples: 
#	1) resnet56_cross_entropy
#	2) resnet56_NLL+MDCA_beta=5.0		(NOTE: SHOULD BE FLOATING POINT)
#	3) resnet56_focal_loss_gamma=1.0 	(NOTE: SHOULD BE FLOATING POINT)
#	4) resnet56_FL+MDCA_gamma=2.0_beta=5.0	(NOTE: SHOULD BE FLOATING POINT)
#	5) resnet56_cross_entropy@resnet56_NLL+MDCA_beta=5.0
#
# --student_model : {model_name_1}@{model_name_2}@ (... so on)
# Examples: resnet8, resnet32, resnet56, resnet8@resnet32, resnet8@resnet32@resnet56
#
# --epochs : It will just accept an integer value.
#
# --dataset : Choose between cifar10/cifar100
################################################################################################################
echo "[Script-message]: You are requested to see the format of inputs to --teacher_model and --student_model in this (KD_script.sh) file."
SHORT=tm:,sm:,d:,e:,h:
LONG=teacher_model:,student_model:,dataset:,epochs:,help
OPTS=$(getopt -a -n scratch --options $SHORT --longoptions $LONG -- "$@")

eval set -- "$OPTS"
echo $OPTS
# If no argument is passed
if [ "$1" == "--" ]; then
	echo "[ERROR]: No argument passed. Expected arguments are: --teacher_model, --student_model, --dataset, --epochs, --help."
	exit 125
fi

# following variables just check if the corresponding argument has been passed
is_teacher_model_there=0
is_student_model_there=0
is_dataset_there=0
is_epochs_there=0

while :
do
  case "$1" in
    --teacher_model )
      teacher_model="$2"
      is_teacher_model_there=1
      shift 2
      ;;
    --student_model )
      student_model="$2"
      is_student_model_there=1
      shift 2
      ;;
    --dataset )
      dataset="$2"
      is_dataset_there=1
      shift 2
      ;;
    --epochs )
      epochs="$2"
      is_epochs_there=1
      shift 2
      ;;
    --help )
      echo "This script trains resnet8, resnet32 and resnet56 models on provided dataset in traditional way (no-KD). Different permutations for \"loss weights\" were used to train models with fancy losses (Non-NLL). Please feel free to set in the commands however you want."
      echo "--teacher_model: Pass in the model type you want to get trained models for."
      echo "--student_model: Pass in the model type you want to train student models for."
      echo "--dataset: Pass in the dataset you want your all models to train on."
      echo "--epochs: Pass in the number of epochs you want to train all your models for."
      echo "Please note: All of the above arguments are compulsory."
      echo "{NOT SO MUCH IMPORTANT}: A note about the script: If you see the script, you might observe, even though I have the same format for the commands for all the three models, i.e., resnet8, resnet32 and resnet56. This is because, sometimes we might want to experiment with different hyperparameters for, let's say, larger model, which is resnet56, over here. That's another reason why I thought, it's better to leave the other detailed hyperparameters in hard code, rather than taking them as argument."
      exit 2
      ;;
    --)
      shift;
      break
      ;;
    *)
      echo "Unexpected option: $1. Expected arguments are: --teacher_model, --student_model, --dataset, --epochs, --help"
      ;;
  esac
done

# Send error if at least one of --model, --dataset and --epochs is not there
if [ $is_teacher_model_there -eq 0 ] || [ $is_student_model_there -eq 0 ] || [ $is_dataset_there -eq 0 ] || [ $is_epochs_there -eq 0 ]; then
	if [ $is_teacher_model_there -eq 0 ]; then
		echo "[ERROR]: --teacher_model is a compulsory argument"
	fi
	if [ $is_student_model_there -eq 0 ]; then
		echo "[ERROR]: --student_model is a compulsory argument"
	fi
	if [ $is_dataset_there -eq 0 ]; then
		echo "[ERROR]: --dataset is a compulsory argument"
	fi
	if [ $is_epochs_there -eq 0 ]; then
		echo "[ERROR]: --epochs is a compulsory argument"
	fi
	exit 125
fi
#####################################################################################################################

########################################################################################################################################################
######################################################## RESNET8 AS STUDENT ############################################################################
########################################################################################################################################################

if [[ "$student_model" == *"resnet8"* ]]; then
	## STUDENT=RESNET8 ; TEACHER=RESNET32_NLL
	if [[ "$teacher_model" == *"resnet32_cross_entropy"* ]]; then
		# The command below just outputs the checkpoint directory of the trained teacher model. In the next line full path to teacher model's checkpoint is created.
		trained_teacher_dirname=`bash find_checkpoint.sh resnet32_cross_entropy ${dataset} `
		trained_teacher_path="trained_model_library/${dataset}/resnet32/${trained_teacher_dirname}"
		CUDA_VISIBLE_DEVICES=0 accelerate launch --mixed_precision fp16 train_student.py \
		--dataset $dataset \
		--model resnet8 \
		--teacher resnet32 \
		--teacher_path $trained_teacher_path \
		--teacher_loss cross_entropy \
		--lr 0.1 \
		--epochs $epochs \
		--wd 1e-4 \
		--train-batch-size 128 \
		--checkpoint distilled_model_library \
		--T 5 --Lambda 0.95
	fi
fi

if [[ "$student_model" == *"resnet8"* ]]; then
	## STUDENT=RESNET8 ; TEACHER=RESNET32_NLL+MDCA
	if [[ "$teacher_model" == *"resnet32_NLL+MDCA"* ]]; then
		# The command below just outputs the checkpoint directory of the trained teacher model. In the next line full path to teacher model's checkpoint is created.
		trained_teacher_dirname=`bash find_checkpoint.sh resnet32_NLL+MDCA_beta=1.0 ${dataset}` # TODO: UPDATE THIS WITH THE BEST beta
		trained_teacher_path="trained_model_library/${dataset}/resnet32/${trained_teacher_dirname}"
		CUDA_VISIBLE_DEVICES=0 accelerate launch --mixed_precision fp16 train_student.py \
		--dataset $dataset \
		--model resnet8 \
		--teacher resnet32 \
		--teacher_path $trained_teacher_path \
		--teacher_loss NLL+MDCA \
		--lr 0.1 \
		--epochs $epochs \
		--wd 1e-4 \
		--train-batch-size 128 \
		--checkpoint distilled_model_library \
		--T 5 --Lambda 0.95
	fi
fi
 
if [[ "$student_model" == *"resnet8"* ]]; then
	## STUDENT=RESNET8 ; TEACHER=RESNET32_FL+MDCA
	if [[ "$teacher_model" == *"resnet32_FL+MDCA"* ]]; then
		# The command below just outputs the checkpoint directory of the trained teacher model. In the next line full path to teacher model's checkpoint is created.
		trained_teacher_dirname=`bash find_checkpoint.sh resnet32_FL+MDCA_gamma=1.0_beta=5.0 ${dataset}` # TODO: UPDATE THIS WITH THE BEST beta & gamma
		trained_teacher_path="trained_model_library/${dataset}/resnet32/${trained_teacher_dirname}"
		CUDA_VISIBLE_DEVICES=0 accelerate launch --mixed_precision fp16 train_student.py \
		--dataset $dataset \
		--model resnet8 \
		--teacher resnet32 \
		--teacher_path $trained_teacher_path \
		--teacher_loss FL+MDCA \
		--lr 0.1 \
		--epochs $epochs \
		--wd 1e-4 \
		--train-batch-size 128 \
		--checkpoint distilled_model_library \
		--T 5 --Lambda 0.95
	fi
fi

if [[ "$student_model" == *"resnet8"* ]]; then
	## STUDENT=RESNET8 ; TEACHER=RESNET32_focal_loss
	if [[ "$teacher_model" == *"resnet32_focal_loss"* ]]; then
		# The command below just outputs the checkpoint directory of the trained teacher model. In the next line full path to teacher model's checkpoint is created.
		trained_teacher_dirname=`bash find_checkpoint.sh resnet32_focal_loss_gamma=1.0 ${dataset}` # TODO: UPDATE THIS WITH THE BEST gamma
		trained_teacher_path="trained_model_library/${dataset}/resnet32/${trained_teacher_dirname}"
		CUDA_VISIBLE_DEVICES=0 accelerate launch --mixed_precision fp16 train_student.py \
		--dataset $dataset \
		--model resnet8 \
		--teacher resnet32 \
		--teacher_path $trained_teacher_path \
		--teacher_loss focal_loss \
		--lr 0.1 \
		--epochs $epochs \
		--wd 1e-4 \
		--train-batch-size 128 \
		--checkpoint distilled_model_library \
		--T 5 --Lambda 0.95
	fi
fi

if [[ "$student_model" == *"resnet8"* ]]; then
	## STUDENT=RESNET8 ; TEACHER=RESNET56_NLL
	if [[ "$teacher_model" == *"resnet56_cross_entropy"* ]]; then
		# The command below just outputs the checkpoint directory of the trained teacher model. In the next line full path to teacher model's checkpoint is created.
		trained_teacher_dirname=`bash find_checkpoint.sh resnet56_cross_entropy ${dataset}`
		trained_teacher_path="trained_model_library/${dataset}/resnet56/${trained_teacher_dirname}"
		CUDA_VISIBLE_DEVICES=0 accelerate launch --mixed_precision fp16 train_student.py \
		--dataset $dataset \
		--model resnet8 \
		--teacher resnet56 \
		--teacher_path $trained_teacher_path \
		--teacher_loss cross_entropy \
		--lr 0.1 \
		--epochs $epochs \
		--wd 1e-4 \
		--train-batch-size 128 \
		--checkpoint distilled_model_library \
		--T 5 --Lambda 0.95
	fi
fi

if [[ "$student_model" == *"resnet8"* ]]; then
	## STUDENT=RESNET8 ; TEACHER=RESNET56_NLL+MDCA
	if [[ "$teacher_model" == *"resnet56_NLL+MDCA"* ]]; then
		# The command below just outputs the checkpoint directory of the trained teacher model. In the next line full path to teacher model's checkpoint is created.
		trained_teacher_dirname=`bash find_checkpoint.sh resnet56_NLL+MDCA_beta=1.0 ${dataset}` # TODO: UPDATE THIS WITH THE BEST beta
		trained_teacher_path="trained_model_library/${dataset}/resnet56/${trained_teacher_dirname}"
		CUDA_VISIBLE_DEVICES=0 accelerate launch --mixed_precision fp16 train_student.py \
		--dataset $dataset \
		--model resnet8 \
		--teacher resnet56 \
		--teacher_path $trained_teacher_path \
		--teacher_loss NLL+MDCA \
		--lr 0.1 \
		--epochs $epochs \
		--wd 1e-4 \
		--train-batch-size 128 \
		--checkpoint distilled_model_library \
		--T 5 --Lambda 0.95
	fi
fi
 
if [[ "$student_model" == *"resnet8"* ]]; then
	## STUDENT=RESNET8 ; TEACHER=RESNET56_FL+MDCA
	if [[ "$teacher_model" == *"resnet56_FL+MDCA"* ]]; then
		# The command below just outputs the checkpoint directory of the trained teacher model. In the next line full path to teacher model's checkpoint is created.
		trained_teacher_dirname=`bash find_checkpoint.sh resnet56_FL+MDCA_gamma=1.0_beta=5.0 ${dataset}` # TODO: UPDATE THIS WITH THE BEST beta & gamma
		trained_teacher_path="trained_model_library/${dataset}/resnet56/${trained_teacher_dirname}"
		CUDA_VISIBLE_DEVICES=0 accelerate launch --mixed_precision fp16 train_student.py \
		--dataset $dataset \
		--model resnet8 \
		--teacher resnet56 \
		--teacher_path $trained_teacher_path \
		--teacher_loss FL+MDCA \
		--lr 0.1 \
		--epochs $epochs \
		--wd 1e-4 \
		--train-batch-size 128 \
		--checkpoint distilled_model_library \
		--T 5 --Lambda 0.95
	fi
fi

if [[ "$student_model" == *"resnet8"* ]]; then
	## STUDENT=RESNET8 ; TEACHER=RESNET56_focal_loss
	if [[ "$teacher_model" == *"resnet56_focal_loss"* ]]; then
		# The command below just outputs the checkpoint directory of the trained teacher model. In the next line full path to teacher model's checkpoint is created.
		trained_teacher_dirname=`bash find_checkpoint.sh resnet56_focal_loss_gamma=1.0 ${dataset}` # TODO: UPDATE THIS WITH THE BEST gamma
		trained_teacher_path="trained_model_library/${dataset}/resnet56/${trained_teacher_dirname}"
		CUDA_VISIBLE_DEVICES=0 accelerate launch --mixed_precision fp16 train_student.py \
		--dataset $dataset \
		--model resnet8 \
		--teacher resnet56 \
		--teacher_path $trained_teacher_path \
		--teacher_loss focal_loss \
		--lr 0.1 \
		--epochs $epochs \
		--wd 1e-4 \
		--train-batch-size 128 \
		--checkpoint distilled_model_library \
		--T 5 --Lambda 0.95
	fi
fi

########################################################################################################################################################
######################################################## RESNET32 AS STUDENT ###########################################################################
########################################################################################################################################################

if [[ "$student_model" == *"resnet32"* ]]; then
	## STUDENT=RESNET32 ; TEACHER=RESNET8_NLL
	if [[ "$teacher_model" == *"resnet8_cross_entropy"* ]]; then
		# The command below just outputs the checkpoint directory of the trained teacher model. In the next line full path to teacher model's checkpoint is created.
		trained_teacher_dirname=`bash find_checkpoint.sh resnet8_cross_entropy ${dataset}`
		trained_teacher_path="trained_model_library/${dataset}/resnet8/${trained_teacher_dirname}"
		CUDA_VISIBLE_DEVICES=0 accelerate launch --mixed_precision fp16 train_student.py \
		--dataset $dataset \
		--model resnet32 \
		--teacher resnet8 \
		--teacher_path $trained_teacher_path \
		--teacher_loss cross_entropy \
		--lr 0.1 \
		--epochs $epochs \
		--wd 1e-4 \
		--train-batch-size 128 \
		--checkpoint distilled_model_library \
		--T 5 --Lambda 0.95
	fi
fi

if [[ "$student_model" == *"resnet32"* ]]; then
	## STUDENT=RESNET32 ; TEACHER=RESNET8_NLL+MDCA
	if [[ "$teacher_model" == *"resnet8_NLL+MDCA"* ]]; then
		# The command below just outputs the checkpoint directory of the trained teacher model. In the next line full path to teacher model's checkpoint is created.
		trained_teacher_dirname=`bash find_checkpoint.sh resnet8_NLL+MDCA_beta=1.0 ${dataset}` # TODO: UPDATE THIS WITH THE BEST beta
		trained_teacher_path="trained_model_library/${dataset}/resnet8/${trained_teacher_dirname}"
		CUDA_VISIBLE_DEVICES=0 accelerate launch --mixed_precision fp16 train_student.py \
		--dataset $dataset \
		--model resnet32 \
		--teacher resnet8 \
		--teacher_path $trained_teacher_path \
		--teacher_loss NLL+MDCA \
		--lr 0.1 \
		--epochs $epochs \
		--wd 1e-4 \
		--train-batch-size 128 \
		--checkpoint distilled_model_library \
		--T 5 --Lambda 0.95
	fi
fi
 
if [[ "$student_model" == *"resnet32"* ]]; then
	## STUDENT=RESNET32 ; TEACHER=RESNET8_FL+MDCA
	if [[ "$teacher_model" == *"resnet8_FL+MDCA"* ]]; then
		# The command below just outputs the checkpoint directory of the trained teacher model. In the next line full path to teacher model's checkpoint is created.
		trained_teacher_dirname=`bash find_checkpoint.sh resnet8_FL+MDCA_gamma=1.0_beta=5.0 ${dataset}` # TODO: UPDATE THIS WITH THE BEST beta & gamma
		trained_teacher_path="trained_model_library/${dataset}/resnet8/${trained_teacher_dirname}"
		CUDA_VISIBLE_DEVICES=0 accelerate launch --mixed_precision fp16 train_student.py \
		--dataset $dataset \
		--model resnet32 \
		--teacher resnet8 \
		--teacher_path $trained_teacher_path \
		--teacher_loss FL+MDCA \
		--lr 0.1 \
		--epochs $epochs \
		--wd 1e-4 \
		--train-batch-size 128 \
		--checkpoint distilled_model_library \
		--T 5 --Lambda 0.95
	fi
fi

if [[ "$student_model" == *"resnet32"* ]]; then
	## STUDENT=RESNET32 ; TEACHER=RESNET8_focal_loss
	if [[ "$teacher_model" == *"resnet8_focal_loss"* ]]; then
		# The command below just outputs the checkpoint directory of the trained teacher model. In the next line full path to teacher model's checkpoint is created.
		trained_teacher_dirname=`bash find_checkpoint.sh resnet8_focal_loss_gamma=1.0 ${dataset}` # TODO: UPDATE THIS WITH THE BEST gamma
		trained_teacher_path="trained_model_library/${dataset}/resnet8/${trained_teacher_dirname}"
		CUDA_VISIBLE_DEVICES=0 accelerate launch --mixed_precision fp16 train_student.py \
		--dataset $dataset \
		--model resnet32 \
		--teacher resnet8 \
		--teacher_path $trained_teacher_path \
		--teacher_loss focal_loss \
		--lr 0.1 \
		--epochs $epochs \
		--wd 1e-4 \
		--train-batch-size 128 \
		--checkpoint distilled_model_library \
		--T 5 --Lambda 0.95
	fi
fi

if [[ "$student_model" == *"resnet32"* ]]; then
	## STUDENT=RESNET32 ; TEACHER=RESNET56_NLL
	if [[ "$teacher_model" == *"resnet56_cross_entropy"* ]]; then
		# The command below just outputs the checkpoint directory of the trained teacher model. In the next line full path to teacher model's checkpoint is created.
		trained_teacher_dirname=`bash find_checkpoint.sh resnet56_cross_entropy ${dataset}`
		trained_teacher_path="trained_model_library/${dataset}/resnet56/${trained_teacher_dirname}"
		CUDA_VISIBLE_DEVICES=0 accelerate launch --mixed_precision fp16 train_student.py \
		--dataset $dataset \
		--model resnet32 \
		--teacher resnet56 \
		--teacher_path $trained_teacher_path \
		--teacher_loss cross_entropy \
		--lr 0.1 \
		--epochs $epochs \
		--wd 1e-4 \
		--train-batch-size 128 \
		--checkpoint distilled_model_library \
		--T 5 --Lambda 0.95
	fi
fi

if [[ "$student_model" == *"resnet32"* ]]; then
	## STUDENT=RESNET32 ; TEACHER=RESNET56_NLL+MDCA
	if [[ "$teacher_model" == *"resnet56_NLL+MDCA"* ]]; then
		# The command below just outputs the checkpoint directory of the trained teacher model. In the next line full path to teacher model's checkpoint is created.
		trained_teacher_dirname=`bash find_checkpoint.sh resnet56_NLL+MDCA_beta=1.0 ${dataset}` # TODO: UPDATE THIS WITH THE BEST beta
		trained_teacher_path="trained_model_library/${dataset}/resnet56/${trained_teacher_dirname}"
		CUDA_VISIBLE_DEVICES=0 accelerate launch --mixed_precision fp16 train_student.py \
		--dataset $dataset \
		--model resnet32 \
		--teacher resnet56 \
		--teacher_path $trained_teacher_path \
		--teacher_loss NLL+MDCA \
		--lr 0.1 \
		--epochs $epochs \
		--wd 1e-4 \
		--train-batch-size 128 \
		--checkpoint distilled_model_library \
		--T 5 --Lambda 0.95
	fi
fi
 
if [[ "$student_model" == *"resnet32"* ]]; then
	## STUDENT=RESNET32 ; TEACHER=RESNET56_FL+MDCA
	if [[ "$teacher_model" == *"resnet56_FL+MDCA"* ]]; then
		# The command below just outputs the checkpoint directory of the trained teacher model. In the next line full path to teacher model's checkpoint is created.
		trained_teacher_dirname=`bash find_checkpoint.sh resnet56_FL+MDCA_gamma=1.0_beta=5.0 ${dataset}` # TODO: UPDATE THIS WITH THE BEST beta & gamma
		trained_teacher_path="trained_model_library/${dataset}/resnet56/${trained_teacher_dirname}"
		CUDA_VISIBLE_DEVICES=0 accelerate launch --mixed_precision fp16 train_student.py \
		--dataset $dataset \
		--model resnet32 \
		--teacher resnet56 \
		--teacher_path $trained_teacher_path \
		--teacher_loss FL+MDCA \
		--lr 0.1 \
		--epochs $epochs \
		--wd 1e-4 \
		--train-batch-size 128 \
		--checkpoint distilled_model_library \
		--T 5 --Lambda 0.95
	fi
fi

if [[ "$student_model" == *"resnet32"* ]]; then
	## STUDENT=RESNET32 ; TEACHER=RESNET56_focal_loss
	if [[ "$teacher_model" == *"resnet56_focal_loss"* ]]; then
		# The command below just outputs the checkpoint directory of the trained teacher model. In the next line full path to teacher model's checkpoint is created.
		trained_teacher_dirname=`bash find_checkpoint.sh resnet56_focal_loss_gamma=1.0 ${dataset}` # TODO: UPDATE THIS WITH THE BEST gamma
		trained_teacher_path="trained_model_library/${dataset}/resnet56/${trained_teacher_dirname}"
		CUDA_VISIBLE_DEVICES=0 accelerate launch --mixed_precision fp16 train_student.py \
		--dataset $dataset \
		--model resnet32 \
		--teacher resnet56 \
		--teacher_path $trained_teacher_path \
		--teacher_loss focal_loss \
		--lr 0.1 \
		--epochs $epochs \
		--wd 1e-4 \
		--train-batch-size 128 \
		--checkpoint distilled_model_library \
		--T 5 --Lambda 0.95
	fi
fi

########################################################################################################################################################
######################################################## RESNET56 AS STUDENT ###########################################################################
########################################################################################################################################################

if [[ "$student_model" == *"resnet56"* ]]; then
	## STUDENT=RESNET56 ; TEACHER=RESNET8_NLL
	if [[ "$teacher_model" == *"resnet8_cross_entropy"* ]]; then
		# The command below just outputs the checkpoint directory of the trained teacher model. In the next line full path to teacher model's checkpoint is created.
		trained_teacher_dirname=`bash find_checkpoint.sh resnet8_cross_entropy ${dataset}`
		trained_teacher_path="trained_model_library/${dataset}/resnet8/${trained_teacher_dirname}"
		CUDA_VISIBLE_DEVICES=0 accelerate launch --mixed_precision fp16 train_student.py \
		--dataset $dataset \
		--model resnet56 \
		--teacher resnet8 \
		--teacher_path $trained_teacher_path \
		--teacher_loss cross_entropy \
		--lr 0.1 \
		--epochs $epochs \
		--wd 1e-4 \
		--train-batch-size 128 \
		--checkpoint distilled_model_library \
		--T 5 --Lambda 0.95
	fi
fi

if [[ "$student_model" == *"resnet56"* ]]; then
	## STUDENT=RESNET56 ; TEACHER=RESNET8_NLL+MDCA
	if [[ "$teacher_model" == *"resnet8_NLL+MDCA"* ]]; then
		# The command below just outputs the checkpoint directory of the trained teacher model. In the next line full path to teacher model's checkpoint is created.
		trained_teacher_dirname=`bash find_checkpoint.sh resnet8_NLL+MDCA_beta=1.0 ${dataset}` # TODO: UPDATE THIS WITH THE BEST beta
		trained_teacher_path="trained_model_library/${dataset}/resnet8/${trained_teacher_dirname}"
		CUDA_VISIBLE_DEVICES=0 accelerate launch --mixed_precision fp16 train_student.py \
		--dataset $dataset \
		--model resnet56 \
		--teacher resnet8 \
		--teacher_path $trained_teacher_path \
		--teacher_loss NLL+MDCA \
		--lr 0.1 \
		--epochs $epochs \
		--wd 1e-4 \
		--train-batch-size 128 \
		--checkpoint distilled_model_library \
		--T 5 --Lambda 0.95
	fi
fi
 
if [[ "$student_model" == *"resnet56"* ]]; then
	## STUDENT=RESNET56 ; TEACHER=RESNET8_FL+MDCA
	if [[ "$teacher_model" == *"resnet8_FL+MDCA"* ]]; then
		# The command below just outputs the checkpoint directory of the trained teacher model. In the next line full path to teacher model's checkpoint is created.
		trained_teacher_dirname=`bash find_checkpoint.sh resnet8_FL+MDCA_gamma=1.0_beta=5.0 ${dataset}` # TODO: UPDATE THIS WITH THE BEST beta & gamma
		trained_teacher_path="trained_model_library/${dataset}/resnet8/${trained_teacher_dirname}"
		CUDA_VISIBLE_DEVICES=0 accelerate launch --mixed_precision fp16 train_student.py \
		--dataset $dataset \
		--model resnet56 \
		--teacher resnet8 \
		--teacher_path $trained_teacher_path \
		--teacher_loss FL+MDCA \
		--lr 0.1 \
		--epochs $epochs \
		--wd 1e-4 \
		--train-batch-size 128 \
		--checkpoint distilled_model_library \
		--T 5 --Lambda 0.95
	fi
fi

if [[ "$student_model" == *"resnet56"* ]]; then
	## STUDENT=RESNET56 ; TEACHER=RESNET8_focal_loss
	if [[ "$teacher_model" == *"resnet8_focal_loss"* ]]; then
		# The command below just outputs the checkpoint directory of the trained teacher model. In the next line full path to teacher model's checkpoint is created.
		trained_teacher_dirname=`bash find_checkpoint.sh resnet8_focal_loss_gamma=1.0 ${dataset}` # TODO: UPDATE THIS WITH THE BEST gamma
		trained_teacher_path="trained_model_library/${dataset}/resnet8/${trained_teacher_dirname}"
		CUDA_VISIBLE_DEVICES=0 accelerate launch --mixed_precision fp16 train_student.py \
		--dataset $dataset \
		--model resnet56 \
		--teacher resnet8 \
		--teacher_path $trained_teacher_path \
		--teacher_loss focal_loss \
		--lr 0.1 \
		--epochs $epochs \
		--wd 1e-4 \
		--train-batch-size 128 \
		--checkpoint distilled_model_library \
		--T 5 --Lambda 0.95
	fi
fi

if [[ "$student_model" == *"resnet56"* ]]; then
	## STUDENT=RESNET56 ; TEACHER=RESNET32_NLL
	if [[ "$teacher_model" == *"resnet32_cross_entropy"* ]]; then
		# The command below just outputs the checkpoint directory of the trained teacher model. In the next line full path to teacher model's checkpoint is created.
		trained_teacher_dirname=`bash find_checkpoint.sh resnet32_cross_entropy ${dataset}`
		trained_teacher_path="trained_model_library/${dataset}/resnet32/${trained_teacher_dirname}"
		CUDA_VISIBLE_DEVICES=0 accelerate launch --mixed_precision fp16 train_student.py \
		--dataset $dataset \
		--model resnet56 \
		--teacher resnet32 \
		--teacher_path $trained_teacher_path \
		--teacher_loss cross_entropy \
		--lr 0.1 \
		--epochs $epochs \
		--wd 1e-4 \
		--train-batch-size 128 \
		--checkpoint distilled_model_library \
		--T 5 --Lambda 0.95
	fi
fi

if [[ "$student_model" == *"resnet56"* ]]; then
	## STUDENT=RESNET56 ; TEACHER=RESNET32_NLL+MDCA
	if [[ "$teacher_model" == *"resnet32_NLL+MDCA"* ]]; then
		# The command below just outputs the checkpoint directory of the trained teacher model. In the next line full path to teacher model's checkpoint is created.
		trained_teacher_dirname=`bash find_checkpoint.sh resnet32_NLL+MDCA_beta=1.0 ${dataset}` # TODO: UPDATE THIS WITH THE BEST beta
		trained_teacher_path="trained_model_library/${dataset}/resnet32/${trained_teacher_dirname}"
		CUDA_VISIBLE_DEVICES=0 accelerate launch --mixed_precision fp16 train_student.py \
		--dataset $dataset \
		--model resnet56 \
		--teacher resnet32 \
		--teacher_path $trained_teacher_path \
		--teacher_loss NLL+MDCA \
		--lr 0.1 \
		--epochs $epochs \
		--wd 1e-4 \
		--train-batch-size 128 \
		--checkpoint distilled_model_library \
		--T 5 --Lambda 0.95
	fi
fi
 
if [[ "$student_model" == *"resnet56"* ]]; then
	## STUDENT=RESNET56 ; TEACHER=RESNET32_FL+MDCA
	if [[ "$teacher_model" == *"resnet32_FL+MDCA"* ]]; then
		# The command below just outputs the checkpoint directory of the trained teacher model. In the next line full path to teacher model's checkpoint is created.
		trained_teacher_dirname=`bash find_checkpoint.sh resnet32_FL+MDCA_gamma=1.0_beta=5.0 ${dataset}` # TODO: UPDATE THIS WITH THE BEST beta & gamma
		trained_teacher_path="trained_model_library/${dataset}/resnet32/${trained_teacher_dirname}"
		CUDA_VISIBLE_DEVICES=0 accelerate launch --mixed_precision fp16 train_student.py \
		--dataset $dataset \
		--model resnet56 \
		--teacher resnet32 \
		--teacher_path $trained_teacher_path \
		--teacher_loss FL+MDCA \
		--lr 0.1 \
		--epochs $epochs \
		--wd 1e-4 \
		--train-batch-size 128 \
		--checkpoint distilled_model_library \
		--T 5 --Lambda 0.95
	fi
fi

if [[ "$student_model" == *"resnet56"* ]]; then
	## STUDENT=RESNET56 ; TEACHER=RESNET32_focal_loss
	if [[ "$teacher_model" == *"resnet32_focal_loss"* ]]; then
		# The command below just outputs the checkpoint directory of the trained teacher model. In the next line full path to teacher model's checkpoint is created.
		trained_teacher_dirname=`bash find_checkpoint.sh resnet32_focal_loss_gamma=1.0 ${dataset}` # TODO: UPDATE THIS WITH THE BEST gamma
		trained_teacher_path="trained_model_library/${dataset}/resnet32/${trained_teacher_dirname}"
		CUDA_VISIBLE_DEVICES=0 accelerate launch --mixed_precision fp16 train_student.py \
		--dataset $dataset \
		--model resnet56 \
		--teacher resnet32 \
		--teacher_path $trained_teacher_path \
		--teacher_loss focal_loss \
		--lr 0.1 \
		--epochs $epochs \
		--wd 1e-4 \
		--train-batch-size 128 \
		--checkpoint distilled_model_library \
		--T 5 --Lambda 0.95
	fi
fi


########################################################################################################################################################
######################################################## CONVNET2 AS STUDENT ###########################################################################
########################################################################################################################################################
## NOTE: Here I'm not hardcoding every permutation for convnet2, 4, 6, 8 & 10, as it might not be necessary.
# Teacher: convnet4
if [[ "$student_model" == *"convnet2"* ]]; then
	## STUDENT=convnet2 ; TEACHER=convnet4_NLL
	if [[ "$teacher_model" == *"convnet4_cross_entropy"* ]]; then
		# The command below just outputs the checkpoint directory of the trained teacher model. In the next line full path to teacher model's checkpoint is created.
		trained_teacher_dirname=`bash find_checkpoint.sh convnet4_cross_entropy ${dataset}`
		trained_teacher_path="trained_model_library/${dataset}/convnet4/${trained_teacher_dirname}"
		CUDA_VISIBLE_DEVICES=0 accelerate launch --mixed_precision fp16 train_student.py \
		--dataset $dataset \
		--model convnet2 \
		--teacher convnet4 \
		--teacher_path $trained_teacher_path \
		--teacher_loss cross_entropy \
		--lr 0.1 \
		--epochs $epochs \
		--wd 1e-4 \
		--train-batch-size 128 \
		--checkpoint distilled_model_library \
		--T 5 --Lambda 0.95
	fi
fi

if [[ "$student_model" == *"convnet2"* ]]; then
	## STUDENT=convnet2 ; TEACHER=convnet4_NLL+MDCA
	if [[ "$teacher_model" == *"convnet4_NLL+MDCA"* ]]; then
		# The command below just outputs the checkpoint directory of the trained teacher model. In the next line full path to teacher model's checkpoint is created.
		trained_teacher_dirname=`bash find_checkpoint.sh convnet4_NLL+MDCA_beta=1.0 ${dataset}` # TODO: UPDATE THIS WITH THE BEST beta
		trained_teacher_path="trained_model_library/${dataset}/convnet4/${trained_teacher_dirname}"
		CUDA_VISIBLE_DEVICES=0 accelerate launch --mixed_precision fp16 train_student.py \
		--dataset $dataset \
		--model convnet2 \
		--teacher convnet4 \
		--teacher_path $trained_teacher_path \
		--teacher_loss NLL+MDCA \
		--lr 0.1 \
		--epochs $epochs \
		--wd 1e-4 \
		--train-batch-size 128 \
		--checkpoint distilled_model_library \
		--T 5 --Lambda 0.95
	fi
fi
 
if [[ "$student_model" == *"convnet2"* ]]; then
	## STUDENT=convnet2 ; TEACHER=convnet4_FL+MDCA
	if [[ "$teacher_model" == *"convnet4_FL+MDCA"* ]]; then
		# The command below just outputs the checkpoint directory of the trained teacher model. In the next line full path to teacher model's checkpoint is created.
		trained_teacher_dirname=`bash find_checkpoint.sh convnet4_FL+MDCA_gamma=1.0_beta=5.0 ${dataset}` # TODO: UPDATE THIS WITH THE BEST beta & gamma
		trained_teacher_path="trained_model_library/${dataset}/convnet4/${trained_teacher_dirname}"
		CUDA_VISIBLE_DEVICES=0 accelerate launch --mixed_precision fp16 train_student.py \
		--dataset $dataset \
		--model convnet2 \
		--teacher convnet4 \
		--teacher_path $trained_teacher_path \
		--teacher_loss FL+MDCA \
		--lr 0.1 \
		--epochs $epochs \
		--wd 1e-4 \
		--train-batch-size 128 \
		--checkpoint distilled_model_library \
		--T 5 --Lambda 0.95
	fi
fi

if [[ "$student_model" == *"convnet2"* ]]; then
	## STUDENT=convnet2 ; TEACHER=convnet4_focal_loss
	if [[ "$teacher_model" == *"convnet4_focal_loss"* ]]; then
		# The command below just outputs the checkpoint directory of the trained teacher model. In the next line full path to teacher model's checkpoint is created.
		trained_teacher_dirname=`bash find_checkpoint.sh convnet4_focal_loss_gamma=1.0 ${dataset}` # TODO: UPDATE THIS WITH THE BEST gamma
		trained_teacher_path="trained_model_library/${dataset}/convnet4/${trained_teacher_dirname}"
		CUDA_VISIBLE_DEVICES=0 accelerate launch --mixed_precision fp16 train_student.py \
		--dataset $dataset \
		--model convnet2 \
		--teacher convnet4 \
		--teacher_path $trained_teacher_path \
		--teacher_loss focal_loss \
		--lr 0.1 \
		--epochs $epochs \
		--wd 1e-4 \
		--train-batch-size 128 \
		--checkpoint distilled_model_library \
		--T 5 --Lambda 0.95
	fi
fi
# Teacher: convnet6
if [[ "$student_model" == *"convnet2"* ]]; then
	## STUDENT=convnet2 ; TEACHER=convnet6_NLL
	if [[ "$teacher_model" == *"convnet6_cross_entropy"* ]]; then
		# The command below just outputs the checkpoint directory of the trained teacher model. In the next line full path to teacher model's checkpoint is created.
		trained_teacher_dirname=`bash find_checkpoint.sh convnet6_cross_entropy ${dataset}`
		trained_teacher_path="trained_model_library/${dataset}/convnet6/${trained_teacher_dirname}"
		CUDA_VISIBLE_DEVICES=0 accelerate launch --mixed_precision fp16 train_student.py \
		--dataset $dataset \
		--model convnet2 \
		--teacher convnet6 \
		--teacher_path $trained_teacher_path \
		--teacher_loss cross_entropy \
		--lr 0.1 \
		--epochs $epochs \
		--wd 1e-4 \
		--train-batch-size 128 \
		--checkpoint distilled_model_library \
		--T 5 --Lambda 0.95
	fi
fi

if [[ "$student_model" == *"convnet2"* ]]; then
	## STUDENT=convnet2 ; TEACHER=convnet6_NLL+MDCA
	if [[ "$teacher_model" == *"convnet6_NLL+MDCA"* ]]; then
		# The command below just outputs the checkpoint directory of the trained teacher model. In the next line full path to teacher model's checkpoint is created.
		trained_teacher_dirname=`bash find_checkpoint.sh convnet6_NLL+MDCA_beta=1.0 ${dataset}` # TODO: UPDATE THIS WITH THE BEST beta
		trained_teacher_path="trained_model_library/${dataset}/convnet6/${trained_teacher_dirname}"
		CUDA_VISIBLE_DEVICES=0 accelerate launch --mixed_precision fp16 train_student.py \
		--dataset $dataset \
		--model convnet2 \
		--teacher convnet6 \
		--teacher_path $trained_teacher_path \
		--teacher_loss NLL+MDCA \
		--lr 0.1 \
		--epochs $epochs \
		--wd 1e-4 \
		--train-batch-size 128 \
		--checkpoint distilled_model_library \
		--T 5 --Lambda 0.95
	fi
fi
 
if [[ "$student_model" == *"convnet2"* ]]; then
	## STUDENT=convnet2 ; TEACHER=convnet6_FL+MDCA
	if [[ "$teacher_model" == *"convnet6_FL+MDCA"* ]]; then
		# The command below just outputs the checkpoint directory of the trained teacher model. In the next line full path to teacher model's checkpoint is created.
		trained_teacher_dirname=`bash find_checkpoint.sh convnet6_FL+MDCA_gamma=1.0_beta=5.0 ${dataset}` # TODO: UPDATE THIS WITH THE BEST beta & gamma
		trained_teacher_path="trained_model_library/${dataset}/convnet6/${trained_teacher_dirname}"
		CUDA_VISIBLE_DEVICES=0 accelerate launch --mixed_precision fp16 train_student.py \
		--dataset $dataset \
		--model convnet2 \
		--teacher convnet6 \
		--teacher_path $trained_teacher_path \
		--teacher_loss FL+MDCA \
		--lr 0.1 \
		--epochs $epochs \
		--wd 1e-4 \
		--train-batch-size 128 \
		--checkpoint distilled_model_library \
		--T 5 --Lambda 0.95
	fi
fi

if [[ "$student_model" == *"convnet2"* ]]; then
	## STUDENT=convnet2 ; TEACHER=convnet6_focal_loss
	if [[ "$teacher_model" == *"convnet6_focal_loss"* ]]; then
		# The command below just outputs the checkpoint directory of the trained teacher model. In the next line full path to teacher model's checkpoint is created.
		trained_teacher_dirname=`bash find_checkpoint.sh convnet6_focal_loss_gamma=1.0 ${dataset}` # TODO: UPDATE THIS WITH THE BEST gamma
		trained_teacher_path="trained_model_library/${dataset}/convnet6/${trained_teacher_dirname}"
		CUDA_VISIBLE_DEVICES=0 accelerate launch --mixed_precision fp16 train_student.py \
		--dataset $dataset \
		--model convnet2 \
		--teacher convnet6 \
		--teacher_path $trained_teacher_path \
		--teacher_loss focal_loss \
		--lr 0.1 \
		--epochs $epochs \
		--wd 1e-4 \
		--train-batch-size 128 \
		--checkpoint distilled_model_library \
		--T 5 --Lambda 0.95
	fi
fi

# Teacher: convnet8
if [[ "$student_model" == *"convnet2"* ]]; then
	## STUDENT=convnet2 ; TEACHER=convnet8_NLL
	if [[ "$teacher_model" == *"convnet8_cross_entropy"* ]]; then
		# The command below just outputs the checkpoint directory of the trained teacher model. In the next line full path to teacher model's checkpoint is created.
		trained_teacher_dirname=`bash find_checkpoint.sh convnet8_cross_entropy ${dataset}`
		trained_teacher_path="trained_model_library/${dataset}/convnet8/${trained_teacher_dirname}"
		CUDA_VISIBLE_DEVICES=0 accelerate launch --mixed_precision fp16 train_student.py \
		--dataset $dataset \
		--model convnet2 \
		--teacher convnet8 \
		--teacher_path $trained_teacher_path \
		--teacher_loss cross_entropy \
		--lr 0.1 \
		--epochs $epochs \
		--wd 1e-4 \
		--train-batch-size 128 \
		--checkpoint distilled_model_library \
		--T 5 --Lambda 0.95
	fi
fi

if [[ "$student_model" == *"convnet2"* ]]; then
	## STUDENT=convnet2 ; TEACHER=convnet8_NLL+MDCA
	if [[ "$teacher_model" == *"convnet8_NLL+MDCA"* ]]; then
		# The command below just outputs the checkpoint directory of the trained teacher model. In the next line full path to teacher model's checkpoint is created.
		trained_teacher_dirname=`bash find_checkpoint.sh convnet8_NLL+MDCA_beta=1.0 ${dataset}` # TODO: UPDATE THIS WITH THE BEST beta
		trained_teacher_path="trained_model_library/${dataset}/convnet8/${trained_teacher_dirname}"
		CUDA_VISIBLE_DEVICES=0 accelerate launch --mixed_precision fp16 train_student.py \
		--dataset $dataset \
		--model convnet2 \
		--teacher convnet8 \
		--teacher_path $trained_teacher_path \
		--teacher_loss NLL+MDCA \
		--lr 0.1 \
		--epochs $epochs \
		--wd 1e-4 \
		--train-batch-size 128 \
		--checkpoint distilled_model_library \
		--T 5 --Lambda 0.95
	fi
fi
 
if [[ "$student_model" == *"convnet2"* ]]; then
	## STUDENT=convnet2 ; TEACHER=convnet8_FL+MDCA
	if [[ "$teacher_model" == *"convnet8_FL+MDCA"* ]]; then
		# The command below just outputs the checkpoint directory of the trained teacher model. In the next line full path to teacher model's checkpoint is created.
		trained_teacher_dirname=`bash find_checkpoint.sh convnet8_FL+MDCA_gamma=1.0_beta=5.0 ${dataset}` # TODO: UPDATE THIS WITH THE BEST beta & gamma
		trained_teacher_path="trained_model_library/${dataset}/convnet8/${trained_teacher_dirname}"
		CUDA_VISIBLE_DEVICES=0 accelerate launch --mixed_precision fp16 train_student.py \
		--dataset $dataset \
		--model convnet2 \
		--teacher convnet8 \
		--teacher_path $trained_teacher_path \
		--teacher_loss FL+MDCA \
		--lr 0.1 \
		--epochs $epochs \
		--wd 1e-4 \
		--train-batch-size 128 \
		--checkpoint distilled_model_library \
		--T 5 --Lambda 0.95
	fi
fi

if [[ "$student_model" == *"convnet2"* ]]; then
	## STUDENT=convnet2 ; TEACHER=convnet8_focal_loss
	if [[ "$teacher_model" == *"convnet8_focal_loss"* ]]; then
		# The command below just outputs the checkpoint directory of the trained teacher model. In the next line full path to teacher model's checkpoint is created.
		trained_teacher_dirname=`bash find_checkpoint.sh convnet8_focal_loss_gamma=1.0 ${dataset}` # TODO: UPDATE THIS WITH THE BEST gamma
		trained_teacher_path="trained_model_library/${dataset}/convnet8/${trained_teacher_dirname}"
		CUDA_VISIBLE_DEVICES=0 accelerate launch --mixed_precision fp16 train_student.py \
		--dataset $dataset \
		--model convnet2 \
		--teacher convnet8 \
		--teacher_path $trained_teacher_path \
		--teacher_loss focal_loss \
		--lr 0.1 \
		--epochs $epochs \
		--wd 1e-4 \
		--train-batch-size 128 \
		--checkpoint distilled_model_library \
		--T 5 --Lambda 0.95
	fi
fi
# Teacher: convnet10
if [[ "$student_model" == *"convnet2"* ]]; then
	## STUDENT=convnet2 ; TEACHER=convnet10_NLL
	if [[ "$teacher_model" == *"convnet10_cross_entropy"* ]]; then
		# The command below just outputs the checkpoint directory of the trained teacher model. In the next line full path to teacher model's checkpoint is created.
		trained_teacher_dirname=`bash find_checkpoint.sh convnet10_cross_entropy ${dataset}`
		trained_teacher_path="trained_model_library/${dataset}/convnet10/${trained_teacher_dirname}"
		CUDA_VISIBLE_DEVICES=0 accelerate launch --mixed_precision fp16 train_student.py \
		--dataset $dataset \
		--model convnet2 \
		--teacher convnet10 \
		--teacher_path $trained_teacher_path \
		--teacher_loss cross_entropy \
		--lr 0.1 \
		--epochs $epochs \
		--wd 1e-4 \
		--train-batch-size 128 \
		--checkpoint distilled_model_library \
		--T 5 --Lambda 0.95
	fi
fi

if [[ "$student_model" == *"convnet2"* ]]; then
	## STUDENT=convnet2 ; TEACHER=convnet10_NLL+MDCA
	if [[ "$teacher_model" == *"convnet10_NLL+MDCA"* ]]; then
		# The command below just outputs the checkpoint directory of the trained teacher model. In the next line full path to teacher model's checkpoint is created.
		trained_teacher_dirname=`bash find_checkpoint.sh convnet10_NLL+MDCA_beta=1.0 ${dataset}` # TODO: UPDATE THIS WITH THE BEST beta
		trained_teacher_path="trained_model_library/${dataset}/convnet10/${trained_teacher_dirname}"
		CUDA_VISIBLE_DEVICES=0 accelerate launch --mixed_precision fp16 train_student.py \
		--dataset $dataset \
		--model convnet2 \
		--teacher convnet10 \
		--teacher_path $trained_teacher_path \
		--teacher_loss NLL+MDCA \
		--lr 0.1 \
		--epochs $epochs \
		--wd 1e-4 \
		--train-batch-size 128 \
		--checkpoint distilled_model_library \
		--T 5 --Lambda 0.95
	fi
fi
 
if [[ "$student_model" == *"convnet2"* ]]; then
	## STUDENT=convnet2 ; TEACHER=convnet10_FL+MDCA
	if [[ "$teacher_model" == *"convnet10_FL+MDCA"* ]]; then
		# The command below just outputs the checkpoint directory of the trained teacher model. In the next line full path to teacher model's checkpoint is created.
		trained_teacher_dirname=`bash find_checkpoint.sh convnet10_FL+MDCA_gamma=1.0_beta=5.0 ${dataset}` # TODO: UPDATE THIS WITH THE BEST beta & gamma
		trained_teacher_path="trained_model_library/${dataset}/convnet10/${trained_teacher_dirname}"
		CUDA_VISIBLE_DEVICES=0 accelerate launch --mixed_precision fp16 train_student.py \
		--dataset $dataset \
		--model convnet2 \
		--teacher convnet10 \
		--teacher_path $trained_teacher_path \
		--teacher_loss FL+MDCA \
		--lr 0.1 \
		--epochs $epochs \
		--wd 1e-4 \
		--train-batch-size 128 \
		--checkpoint distilled_model_library \
		--T 5 --Lambda 0.95
	fi
fi

if [[ "$student_model" == *"convnet2"* ]]; then
	## STUDENT=convnet2 ; TEACHER=convnet10_focal_loss
	if [[ "$teacher_model" == *"convnet10_focal_loss"* ]]; then
		# The command below just outputs the checkpoint directory of the trained teacher model. In the next line full path to teacher model's checkpoint is created.
		trained_teacher_dirname=`bash find_checkpoint.sh convnet10_focal_loss_gamma=1.0 ${dataset}` # TODO: UPDATE THIS WITH THE BEST gamma
		trained_teacher_path="trained_model_library/${dataset}/convnet10/${trained_teacher_dirname}"
		CUDA_VISIBLE_DEVICES=0 accelerate launch --mixed_precision fp16 train_student.py \
		--dataset $dataset \
		--model convnet2 \
		--teacher convnet10 \
		--teacher_path $trained_teacher_path \
		--teacher_loss focal_loss \
		--lr 0.1 \
		--epochs $epochs \
		--wd 1e-4 \
		--train-batch-size 128 \
		--checkpoint distilled_model_library \
		--T 5 --Lambda 0.95
	fi
fi

########################################################################################################################################################
######################################################## CONVNET4 AS STUDENT ###########################################################################
########################################################################################################################################################

# Teacher: convnet2
if [[ "$student_model" == *"convnet4"* ]]; then
	## STUDENT=convnet4 ; TEACHER=convnet2_NLL
	if [[ "$teacher_model" == *"convnet2_cross_entropy"* ]]; then
		# The command below just outputs the checkpoint directory of the trained teacher model. In the next line full path to teacher model's checkpoint is created.
		trained_teacher_dirname=`bash find_checkpoint.sh convnet2_cross_entropy ${dataset}`
		trained_teacher_path="trained_model_library/${dataset}/convnet2/${trained_teacher_dirname}"
		CUDA_VISIBLE_DEVICES=0 accelerate launch --mixed_precision fp16 train_student.py \
		--dataset $dataset \
		--model convnet4 \
		--teacher convnet2 \
		--teacher_path $trained_teacher_path \
		--teacher_loss cross_entropy \
		--lr 0.1 \
		--epochs $epochs \
		--wd 1e-4 \
		--train-batch-size 128 \
		--checkpoint distilled_model_library \
		--T 5 --Lambda 0.95
	fi
fi

if [[ "$student_model" == *"convnet4"* ]]; then
	## STUDENT=convnet4 ; TEACHER=convnet2_NLL+MDCA
	if [[ "$teacher_model" == *"convnet2_NLL+MDCA"* ]]; then
		# The command below just outputs the checkpoint directory of the trained teacher model. In the next line full path to teacher model's checkpoint is created.
		trained_teacher_dirname=`bash find_checkpoint.sh convnet2_NLL+MDCA_beta=1.0 ${dataset}` # TODO: UPDATE THIS WITH THE BEST beta
		trained_teacher_path="trained_model_library/${dataset}/convnet2/${trained_teacher_dirname}"
		CUDA_VISIBLE_DEVICES=0 accelerate launch --mixed_precision fp16 train_student.py \
		--dataset $dataset \
		--model convnet4 \
		--teacher convnet2 \
		--teacher_path $trained_teacher_path \
		--teacher_loss NLL+MDCA \
		--lr 0.1 \
		--epochs $epochs \
		--wd 1e-4 \
		--train-batch-size 128 \
		--checkpoint distilled_model_library \
		--T 5 --Lambda 0.95
	fi
fi
 
if [[ "$student_model" == *"convnet4"* ]]; then
	## STUDENT=convnet4 ; TEACHER=convnet2_FL+MDCA
	if [[ "$teacher_model" == *"convnet2_FL+MDCA"* ]]; then
		# The command below just outputs the checkpoint directory of the trained teacher model. In the next line full path to teacher model's checkpoint is created.
		trained_teacher_dirname=`bash find_checkpoint.sh convnet2_FL+MDCA_gamma=1.0_beta=5.0 ${dataset}` # TODO: UPDATE THIS WITH THE BEST beta & gamma
		trained_teacher_path="trained_model_library/${dataset}/convnet2/${trained_teacher_dirname}"
		CUDA_VISIBLE_DEVICES=0 accelerate launch --mixed_precision fp16 train_student.py \
		--dataset $dataset \
		--model convnet4 \
		--teacher convnet2 \
		--teacher_path $trained_teacher_path \
		--teacher_loss FL+MDCA \
		--lr 0.1 \
		--epochs $epochs \
		--wd 1e-4 \
		--train-batch-size 128 \
		--checkpoint distilled_model_library \
		--T 5 --Lambda 0.95
	fi
fi

if [[ "$student_model" == *"convnet4"* ]]; then
	## STUDENT=convnet4 ; TEACHER=convnet2_focal_loss
	if [[ "$teacher_model" == *"convnet2_focal_loss"* ]]; then
		# The command below just outputs the checkpoint directory of the trained teacher model. In the next line full path to teacher model's checkpoint is created.
		trained_teacher_dirname=`bash find_checkpoint.sh convnet2_focal_loss_gamma=1.0 ${dataset}` # TODO: UPDATE THIS WITH THE BEST gamma
		trained_teacher_path="trained_model_library/${dataset}/convnet2/${trained_teacher_dirname}"
		CUDA_VISIBLE_DEVICES=0 accelerate launch --mixed_precision fp16 train_student.py \
		--dataset $dataset \
		--model convnet4 \
		--teacher convnet2 \
		--teacher_path $trained_teacher_path \
		--teacher_loss focal_loss \
		--lr 0.1 \
		--epochs $epochs \
		--wd 1e-4 \
		--train-batch-size 128 \
		--checkpoint distilled_model_library \
		--T 5 --Lambda 0.95
	fi
fi

# Teacher: convnet6
if [[ "$student_model" == *"convnet4"* ]]; then
	## STUDENT=convnet4 ; TEACHER=convnet6_NLL
	if [[ "$teacher_model" == *"convnet6_cross_entropy"* ]]; then
		# The command below just outputs the checkpoint directory of the trained teacher model. In the next line full path to teacher model's checkpoint is created.
		trained_teacher_dirname=`bash find_checkpoint.sh convnet6_cross_entropy ${dataset}`
		trained_teacher_path="trained_model_library/${dataset}/convnet6/${trained_teacher_dirname}"
		CUDA_VISIBLE_DEVICES=0 accelerate launch --mixed_precision fp16 train_student.py \
		--dataset $dataset \
		--model convnet4 \
		--teacher convnet6 \
		--teacher_path $trained_teacher_path \
		--teacher_loss cross_entropy \
		--lr 0.1 \
		--epochs $epochs \
		--wd 1e-4 \
		--train-batch-size 128 \
		--checkpoint distilled_model_library \
		--T 5 --Lambda 0.95
	fi
fi

if [[ "$student_model" == *"convnet4"* ]]; then
	## STUDENT=convnet4 ; TEACHER=convnet6_NLL+MDCA
	if [[ "$teacher_model" == *"convnet6_NLL+MDCA"* ]]; then
		# The command below just outputs the checkpoint directory of the trained teacher model. In the next line full path to teacher model's checkpoint is created.
		trained_teacher_dirname=`bash find_checkpoint.sh convnet6_NLL+MDCA_beta=1.0 ${dataset}` # TODO: UPDATE THIS WITH THE BEST beta
		trained_teacher_path="trained_model_library/${dataset}/convnet6/${trained_teacher_dirname}"
		CUDA_VISIBLE_DEVICES=0 accelerate launch --mixed_precision fp16 train_student.py \
		--dataset $dataset \
		--model convnet4 \
		--teacher convnet6 \
		--teacher_path $trained_teacher_path \
		--teacher_loss NLL+MDCA \
		--lr 0.1 \
		--epochs $epochs \
		--wd 1e-4 \
		--train-batch-size 128 \
		--checkpoint distilled_model_library \
		--T 5 --Lambda 0.95
	fi
fi
 
if [[ "$student_model" == *"convnet4"* ]]; then
	## STUDENT=convnet4 ; TEACHER=convnet6_FL+MDCA
	if [[ "$teacher_model" == *"convnet6_FL+MDCA"* ]]; then
		# The command below just outputs the checkpoint directory of the trained teacher model. In the next line full path to teacher model's checkpoint is created.
		trained_teacher_dirname=`bash find_checkpoint.sh convnet6_FL+MDCA_gamma=1.0_beta=5.0 ${dataset}` # TODO: UPDATE THIS WITH THE BEST beta & gamma
		trained_teacher_path="trained_model_library/${dataset}/convnet6/${trained_teacher_dirname}"
		CUDA_VISIBLE_DEVICES=0 accelerate launch --mixed_precision fp16 train_student.py \
		--dataset $dataset \
		--model convnet4 \
		--teacher convnet6 \
		--teacher_path $trained_teacher_path \
		--teacher_loss FL+MDCA \
		--lr 0.1 \
		--epochs $epochs \
		--wd 1e-4 \
		--train-batch-size 128 \
		--checkpoint distilled_model_library \
		--T 5 --Lambda 0.95
	fi
fi

if [[ "$student_model" == *"convnet4"* ]]; then
	## STUDENT=convnet4 ; TEACHER=convnet6_focal_loss
	if [[ "$teacher_model" == *"convnet6_focal_loss"* ]]; then
		# The command below just outputs the checkpoint directory of the trained teacher model. In the next line full path to teacher model's checkpoint is created.
		trained_teacher_dirname=`bash find_checkpoint.sh convnet6_focal_loss_gamma=1.0 ${dataset}` # TODO: UPDATE THIS WITH THE BEST gamma
		trained_teacher_path="trained_model_library/${dataset}/convnet6/${trained_teacher_dirname}"
		CUDA_VISIBLE_DEVICES=0 accelerate launch --mixed_precision fp16 train_student.py \
		--dataset $dataset \
		--model convnet4 \
		--teacher convnet6 \
		--teacher_path $trained_teacher_path \
		--teacher_loss focal_loss \
		--lr 0.1 \
		--epochs $epochs \
		--wd 1e-4 \
		--train-batch-size 128 \
		--checkpoint distilled_model_library \
		--T 5 --Lambda 0.95
	fi
fi

# Teacher: convnet8
if [[ "$student_model" == *"convnet4"* ]]; then
	## STUDENT=convnet4 ; TEACHER=convnet8_NLL
	if [[ "$teacher_model" == *"convnet8_cross_entropy"* ]]; then
		# The command below just outputs the checkpoint directory of the trained teacher model. In the next line full path to teacher model's checkpoint is created.
		trained_teacher_dirname=`bash find_checkpoint.sh convnet8_cross_entropy ${dataset}`
		trained_teacher_path="trained_model_library/${dataset}/convnet8/${trained_teacher_dirname}"
		CUDA_VISIBLE_DEVICES=0 accelerate launch --mixed_precision fp16 train_student.py \
		--dataset $dataset \
		--model convnet4 \
		--teacher convnet8 \
		--teacher_path $trained_teacher_path \
		--teacher_loss cross_entropy \
		--lr 0.1 \
		--epochs $epochs \
		--wd 1e-4 \
		--train-batch-size 128 \
		--checkpoint distilled_model_library \
		--T 5 --Lambda 0.95
	fi
fi

if [[ "$student_model" == *"convnet4"* ]]; then
	## STUDENT=convnet4 ; TEACHER=convnet8_NLL+MDCA
	if [[ "$teacher_model" == *"convnet8_NLL+MDCA"* ]]; then
		# The command below just outputs the checkpoint directory of the trained teacher model. In the next line full path to teacher model's checkpoint is created.
		trained_teacher_dirname=`bash find_checkpoint.sh convnet8_NLL+MDCA_beta=1.0 ${dataset}` # TODO: UPDATE THIS WITH THE BEST beta
		trained_teacher_path="trained_model_library/${dataset}/convnet8/${trained_teacher_dirname}"
		CUDA_VISIBLE_DEVICES=0 accelerate launch --mixed_precision fp16 train_student.py \
		--dataset $dataset \
		--model convnet4 \
		--teacher convnet8 \
		--teacher_path $trained_teacher_path \
		--teacher_loss NLL+MDCA \
		--lr 0.1 \
		--epochs $epochs \
		--wd 1e-4 \
		--train-batch-size 128 \
		--checkpoint distilled_model_library \
		--T 5 --Lambda 0.95
	fi
fi
 
if [[ "$student_model" == *"convnet4"* ]]; then
	## STUDENT=convnet4 ; TEACHER=convnet8_FL+MDCA
	if [[ "$teacher_model" == *"convnet8_FL+MDCA"* ]]; then
		# The command below just outputs the checkpoint directory of the trained teacher model. In the next line full path to teacher model's checkpoint is created.
		trained_teacher_dirname=`bash find_checkpoint.sh convnet8_FL+MDCA_gamma=1.0_beta=5.0 ${dataset}` # TODO: UPDATE THIS WITH THE BEST beta & gamma
		trained_teacher_path="trained_model_library/${dataset}/convnet8/${trained_teacher_dirname}"
		CUDA_VISIBLE_DEVICES=0 accelerate launch --mixed_precision fp16 train_student.py \
		--dataset $dataset \
		--model convnet4 \
		--teacher convnet8 \
		--teacher_path $trained_teacher_path \
		--teacher_loss FL+MDCA \
		--lr 0.1 \
		--epochs $epochs \
		--wd 1e-4 \
		--train-batch-size 128 \
		--checkpoint distilled_model_library \
		--T 5 --Lambda 0.95
	fi
fi

if [[ "$student_model" == *"convnet4"* ]]; then
	## STUDENT=convnet4 ; TEACHER=convnet8_focal_loss
	if [[ "$teacher_model" == *"convnet8_focal_loss"* ]]; then
		# The command below just outputs the checkpoint directory of the trained teacher model. In the next line full path to teacher model's checkpoint is created.
		trained_teacher_dirname=`bash find_checkpoint.sh convnet8_focal_loss_gamma=1.0 ${dataset}` # TODO: UPDATE THIS WITH THE BEST gamma
		trained_teacher_path="trained_model_library/${dataset}/convnet8/${trained_teacher_dirname}"
		CUDA_VISIBLE_DEVICES=0 accelerate launch --mixed_precision fp16 train_student.py \
		--dataset $dataset \
		--model convnet4 \
		--teacher convnet8 \
		--teacher_path $trained_teacher_path \
		--teacher_loss focal_loss \
		--lr 0.1 \
		--epochs $epochs \
		--wd 1e-4 \
		--train-batch-size 128 \
		--checkpoint distilled_model_library \
		--T 5 --Lambda 0.95
	fi
fi

# Teacher: convnet10
if [[ "$student_model" == *"convnet4"* ]]; then
	## STUDENT=convnet4 ; TEACHER=convnet10_NLL
	if [[ "$teacher_model" == *"convnet10_cross_entropy"* ]]; then
		# The command below just outputs the checkpoint directory of the trained teacher model. In the next line full path to teacher model's checkpoint is created.
		trained_teacher_dirname=`bash find_checkpoint.sh convnet10_cross_entropy ${dataset}`
		trained_teacher_path="trained_model_library/${dataset}/convnet10/${trained_teacher_dirname}"
		CUDA_VISIBLE_DEVICES=0 accelerate launch --mixed_precision fp16 train_student.py \
		--dataset $dataset \
		--model convnet4 \
		--teacher convnet10 \
		--teacher_path $trained_teacher_path \
		--teacher_loss cross_entropy \
		--lr 0.1 \
		--epochs $epochs \
		--wd 1e-4 \
		--train-batch-size 128 \
		--checkpoint distilled_model_library \
		--T 5 --Lambda 0.95
	fi
fi

if [[ "$student_model" == *"convnet4"* ]]; then
	## STUDENT=convnet4 ; TEACHER=convnet10_NLL+MDCA
	if [[ "$teacher_model" == *"convnet10_NLL+MDCA"* ]]; then
		# The command below just outputs the checkpoint directory of the trained teacher model. In the next line full path to teacher model's checkpoint is created.
		trained_teacher_dirname=`bash find_checkpoint.sh convnet10_NLL+MDCA_beta=1.0 ${dataset}` # TODO: UPDATE THIS WITH THE BEST beta
		trained_teacher_path="trained_model_library/${dataset}/convnet10/${trained_teacher_dirname}"
		CUDA_VISIBLE_DEVICES=0 accelerate launch --mixed_precision fp16 train_student.py \
		--dataset $dataset \
		--model convnet4 \
		--teacher convnet10 \
		--teacher_path $trained_teacher_path \
		--teacher_loss NLL+MDCA \
		--lr 0.1 \
		--epochs $epochs \
		--wd 1e-4 \
		--train-batch-size 128 \
		--checkpoint distilled_model_library \
		--T 5 --Lambda 0.95
	fi
fi
 
if [[ "$student_model" == *"convnet4"* ]]; then
	## STUDENT=convnet4 ; TEACHER=convnet10_FL+MDCA
	if [[ "$teacher_model" == *"convnet10_FL+MDCA"* ]]; then
		# The command below just outputs the checkpoint directory of the trained teacher model. In the next line full path to teacher model's checkpoint is created.
		trained_teacher_dirname=`bash find_checkpoint.sh convnet10_FL+MDCA_gamma=1.0_beta=5.0 ${dataset}` # TODO: UPDATE THIS WITH THE BEST beta & gamma
		trained_teacher_path="trained_model_library/${dataset}/convnet10/${trained_teacher_dirname}"
		CUDA_VISIBLE_DEVICES=0 accelerate launch --mixed_precision fp16 train_student.py \
		--dataset $dataset \
		--model convnet4 \
		--teacher convnet10 \
		--teacher_path $trained_teacher_path \
		--teacher_loss FL+MDCA \
		--lr 0.1 \
		--epochs $epochs \
		--wd 1e-4 \
		--train-batch-size 128 \
		--checkpoint distilled_model_library \
		--T 5 --Lambda 0.95
	fi
fi

if [[ "$student_model" == *"convnet4"* ]]; then
	## STUDENT=convnet4 ; TEACHER=convnet10_focal_loss
	if [[ "$teacher_model" == *"convnet10_focal_loss"* ]]; then
		# The command below just outputs the checkpoint directory of the trained teacher model. In the next line full path to teacher model's checkpoint is created.
		trained_teacher_dirname=`bash find_checkpoint.sh convnet10_focal_loss_gamma=1.0 ${dataset}` # TODO: UPDATE THIS WITH THE BEST gamma
		trained_teacher_path="trained_model_library/${dataset}/convnet10/${trained_teacher_dirname}"
		CUDA_VISIBLE_DEVICES=0 accelerate launch --mixed_precision fp16 train_student.py \
		--dataset $dataset \
		--model convnet4 \
		--teacher convnet10 \
		--teacher_path $trained_teacher_path \
		--teacher_loss focal_loss \
		--lr 0.1 \
		--epochs $epochs \
		--wd 1e-4 \
		--train-batch-size 128 \
		--checkpoint distilled_model_library \
		--T 5 --Lambda 0.95
	fi
fi

########################################################################################################################################################
######################################################## CONVNET6 AS STUDENT ###########################################################################
########################################################################################################################################################

# Teacher: convnet2
if [[ "$student_model" == *"convnet6"* ]]; then
	## STUDENT=convnet6 ; TEACHER=convnet2_NLL
	if [[ "$teacher_model" == *"convnet2_cross_entropy"* ]]; then
		# The command below just outputs the checkpoint directory of the trained teacher model. In the next line full path to teacher model's checkpoint is created.
		trained_teacher_dirname=`bash find_checkpoint.sh convnet2_cross_entropy ${dataset}`
		trained_teacher_path="trained_model_library/${dataset}/convnet2/${trained_teacher_dirname}"
		CUDA_VISIBLE_DEVICES=0 accelerate launch --mixed_precision fp16 train_student.py \
		--dataset $dataset \
		--model convnet6 \
		--teacher convnet2 \
		--teacher_path $trained_teacher_path \
		--teacher_loss cross_entropy \
		--lr 0.1 \
		--epochs $epochs \
		--wd 1e-4 \
		--train-batch-size 128 \
		--checkpoint distilled_model_library \
		--T 5 --Lambda 0.95
	fi
fi

if [[ "$student_model" == *"convnet6"* ]]; then
	## STUDENT=convnet6 ; TEACHER=convnet2_NLL+MDCA
	if [[ "$teacher_model" == *"convnet2_NLL+MDCA"* ]]; then
		# The command below just outputs the checkpoint directory of the trained teacher model. In the next line full path to teacher model's checkpoint is created.
		trained_teacher_dirname=`bash find_checkpoint.sh convnet2_NLL+MDCA_beta=1.0 ${dataset}` # TODO: UPDATE THIS WITH THE BEST beta
		trained_teacher_path="trained_model_library/${dataset}/convnet2/${trained_teacher_dirname}"
		CUDA_VISIBLE_DEVICES=0 accelerate launch --mixed_precision fp16 train_student.py \
		--dataset $dataset \
		--model convnet6 \
		--teacher convnet2 \
		--teacher_path $trained_teacher_path \
		--teacher_loss NLL+MDCA \
		--lr 0.1 \
		--epochs $epochs \
		--wd 1e-4 \
		--train-batch-size 128 \
		--checkpoint distilled_model_library \
		--T 5 --Lambda 0.95
	fi
fi
 
if [[ "$student_model" == *"convnet6"* ]]; then
	## STUDENT=convnet6 ; TEACHER=convnet2_FL+MDCA
	if [[ "$teacher_model" == *"convnet2_FL+MDCA"* ]]; then
		# The command below just outputs the checkpoint directory of the trained teacher model. In the next line full path to teacher model's checkpoint is created.
		trained_teacher_dirname=`bash find_checkpoint.sh convnet2_FL+MDCA_gamma=1.0_beta=5.0 ${dataset}` # TODO: UPDATE THIS WITH THE BEST beta & gamma
		trained_teacher_path="trained_model_library/${dataset}/convnet2/${trained_teacher_dirname}"
		CUDA_VISIBLE_DEVICES=0 accelerate launch --mixed_precision fp16 train_student.py \
		--dataset $dataset \
		--model convnet6 \
		--teacher convnet2 \
		--teacher_path $trained_teacher_path \
		--teacher_loss FL+MDCA \
		--lr 0.1 \
		--epochs $epochs \
		--wd 1e-4 \
		--train-batch-size 128 \
		--checkpoint distilled_model_library \
		--T 5 --Lambda 0.95
	fi
fi

if [[ "$student_model" == *"convnet6"* ]]; then
	## STUDENT=convnet6 ; TEACHER=convnet2_focal_loss
	if [[ "$teacher_model" == *"convnet2_focal_loss"* ]]; then
		# The command below just outputs the checkpoint directory of the trained teacher model. In the next line full path to teacher model's checkpoint is created.
		trained_teacher_dirname=`bash find_checkpoint.sh convnet2_focal_loss_gamma=1.0 ${dataset}` # TODO: UPDATE THIS WITH THE BEST gamma
		trained_teacher_path="trained_model_library/${dataset}/convnet2/${trained_teacher_dirname}"
		CUDA_VISIBLE_DEVICES=0 accelerate launch --mixed_precision fp16 train_student.py \
		--dataset $dataset \
		--model convnet6 \
		--teacher convnet2 \
		--teacher_path $trained_teacher_path \
		--teacher_loss focal_loss \
		--lr 0.1 \
		--epochs $epochs \
		--wd 1e-4 \
		--train-batch-size 128 \
		--checkpoint distilled_model_library \
		--T 5 --Lambda 0.95
	fi
fi
# Teacher: convnet4
if [[ "$student_model" == *"convnet6"* ]]; then
	## STUDENT=convnet6 ; TEACHER=convnet4_NLL
	if [[ "$teacher_model" == *"convnet4_cross_entropy"* ]]; then
		# The command below just outputs the checkpoint directory of the trained teacher model. In the next line full path to teacher model's checkpoint is created.
		trained_teacher_dirname=`bash find_checkpoint.sh convnet4_cross_entropy ${dataset}`
		trained_teacher_path="trained_model_library/${dataset}/convnet4/${trained_teacher_dirname}"
		CUDA_VISIBLE_DEVICES=0 accelerate launch --mixed_precision fp16 train_student.py \
		--dataset $dataset \
		--model convnet6 \
		--teacher convnet4 \
		--teacher_path $trained_teacher_path \
		--teacher_loss cross_entropy \
		--lr 0.1 \
		--epochs $epochs \
		--wd 1e-4 \
		--train-batch-size 128 \
		--checkpoint distilled_model_library \
		--T 5 --Lambda 0.95
	fi
fi

if [[ "$student_model" == *"convnet6"* ]]; then
	## STUDENT=convnet6 ; TEACHER=convnet4_NLL+MDCA
	if [[ "$teacher_model" == *"convnet4_NLL+MDCA"* ]]; then
		# The command below just outputs the checkpoint directory of the trained teacher model. In the next line full path to teacher model's checkpoint is created.
		trained_teacher_dirname=`bash find_checkpoint.sh convnet4_NLL+MDCA_beta=1.0 ${dataset}` # TODO: UPDATE THIS WITH THE BEST beta
		trained_teacher_path="trained_model_library/${dataset}/convnet4/${trained_teacher_dirname}"
		CUDA_VISIBLE_DEVICES=0 accelerate launch --mixed_precision fp16 train_student.py \
		--dataset $dataset \
		--model convnet6 \
		--teacher convnet4 \
		--teacher_path $trained_teacher_path \
		--teacher_loss NLL+MDCA \
		--lr 0.1 \
		--epochs $epochs \
		--wd 1e-4 \
		--train-batch-size 128 \
		--checkpoint distilled_model_library \
		--T 5 --Lambda 0.95
	fi
fi
 
if [[ "$student_model" == *"convnet6"* ]]; then
	## STUDENT=convnet6 ; TEACHER=convnet4_FL+MDCA
	if [[ "$teacher_model" == *"convnet4_FL+MDCA"* ]]; then
		# The command below just outputs the checkpoint directory of the trained teacher model. In the next line full path to teacher model's checkpoint is created.
		trained_teacher_dirname=`bash find_checkpoint.sh convnet4_FL+MDCA_gamma=1.0_beta=5.0 ${dataset}` # TODO: UPDATE THIS WITH THE BEST beta & gamma
		trained_teacher_path="trained_model_library/${dataset}/convnet4/${trained_teacher_dirname}"
		CUDA_VISIBLE_DEVICES=0 accelerate launch --mixed_precision fp16 train_student.py \
		--dataset $dataset \
		--model convnet6 \
		--teacher convnet4 \
		--teacher_path $trained_teacher_path \
		--teacher_loss FL+MDCA \
		--lr 0.1 \
		--epochs $epochs \
		--wd 1e-4 \
		--train-batch-size 128 \
		--checkpoint distilled_model_library \
		--T 5 --Lambda 0.95
	fi
fi

if [[ "$student_model" == *"convnet6"* ]]; then
	## STUDENT=convnet6 ; TEACHER=convnet4_focal_loss
	if [[ "$teacher_model" == *"convnet4_focal_loss"* ]]; then
		# The command below just outputs the checkpoint directory of the trained teacher model. In the next line full path to teacher model's checkpoint is created.
		trained_teacher_dirname=`bash find_checkpoint.sh convnet4_focal_loss_gamma=1.0 ${dataset}` # TODO: UPDATE THIS WITH THE BEST gamma
		trained_teacher_path="trained_model_library/${dataset}/convnet4/${trained_teacher_dirname}"
		CUDA_VISIBLE_DEVICES=0 accelerate launch --mixed_precision fp16 train_student.py \
		--dataset $dataset \
		--model convnet6 \
		--teacher convnet4 \
		--teacher_path $trained_teacher_path \
		--teacher_loss focal_loss \
		--lr 0.1 \
		--epochs $epochs \
		--wd 1e-4 \
		--train-batch-size 128 \
		--checkpoint distilled_model_library \
		--T 5 --Lambda 0.95
	fi
fi

# Teacher: convnet8
if [[ "$student_model" == *"convnet6"* ]]; then
	## STUDENT=convnet6 ; TEACHER=convnet8_NLL
	if [[ "$teacher_model" == *"convnet8_cross_entropy"* ]]; then
		# The command below just outputs the checkpoint directory of the trained teacher model. In the next line full path to teacher model's checkpoint is created.
		trained_teacher_dirname=`bash find_checkpoint.sh convnet8_cross_entropy ${dataset}`
		trained_teacher_path="trained_model_library/${dataset}/convnet8/${trained_teacher_dirname}"
		CUDA_VISIBLE_DEVICES=0 accelerate launch --mixed_precision fp16 train_student.py \
		--dataset $dataset \
		--model convnet6 \
		--teacher convnet8 \
		--teacher_path $trained_teacher_path \
		--teacher_loss cross_entropy \
		--lr 0.1 \
		--epochs $epochs \
		--wd 1e-4 \
		--train-batch-size 128 \
		--checkpoint distilled_model_library \
		--T 5 --Lambda 0.95
	fi
fi

if [[ "$student_model" == *"convnet6"* ]]; then
	## STUDENT=convnet6 ; TEACHER=convnet8_NLL+MDCA
	if [[ "$teacher_model" == *"convnet8_NLL+MDCA"* ]]; then
		# The command below just outputs the checkpoint directory of the trained teacher model. In the next line full path to teacher model's checkpoint is created.
		trained_teacher_dirname=`bash find_checkpoint.sh convnet8_NLL+MDCA_beta=1.0 ${dataset}` # TODO: UPDATE THIS WITH THE BEST beta
		trained_teacher_path="trained_model_library/${dataset}/convnet8/${trained_teacher_dirname}"
		CUDA_VISIBLE_DEVICES=0 accelerate launch --mixed_precision fp16 train_student.py \
		--dataset $dataset \
		--model convnet6 \
		--teacher convnet8 \
		--teacher_path $trained_teacher_path \
		--teacher_loss NLL+MDCA \
		--lr 0.1 \
		--epochs $epochs \
		--wd 1e-4 \
		--train-batch-size 128 \
		--checkpoint distilled_model_library \
		--T 5 --Lambda 0.95
	fi
fi
 
if [[ "$student_model" == *"convnet6"* ]]; then
	## STUDENT=convnet6 ; TEACHER=convnet8_FL+MDCA
	if [[ "$teacher_model" == *"convnet8_FL+MDCA"* ]]; then
		# The command below just outputs the checkpoint directory of the trained teacher model. In the next line full path to teacher model's checkpoint is created.
		trained_teacher_dirname=`bash find_checkpoint.sh convnet8_FL+MDCA_gamma=1.0_beta=5.0 ${dataset}` # TODO: UPDATE THIS WITH THE BEST beta & gamma
		trained_teacher_path="trained_model_library/${dataset}/convnet8/${trained_teacher_dirname}"
		CUDA_VISIBLE_DEVICES=0 accelerate launch --mixed_precision fp16 train_student.py \
		--dataset $dataset \
		--model convnet6 \
		--teacher convnet8 \
		--teacher_path $trained_teacher_path \
		--teacher_loss FL+MDCA \
		--lr 0.1 \
		--epochs $epochs \
		--wd 1e-4 \
		--train-batch-size 128 \
		--checkpoint distilled_model_library \
		--T 5 --Lambda 0.95
	fi
fi

if [[ "$student_model" == *"convnet6"* ]]; then
	## STUDENT=convnet6 ; TEACHER=convnet8_focal_loss
	if [[ "$teacher_model" == *"convnet8_focal_loss"* ]]; then
		# The command below just outputs the checkpoint directory of the trained teacher model. In the next line full path to teacher model's checkpoint is created.
		trained_teacher_dirname=`bash find_checkpoint.sh convnet8_focal_loss_gamma=1.0 ${dataset}` # TODO: UPDATE THIS WITH THE BEST gamma
		trained_teacher_path="trained_model_library/${dataset}/convnet8/${trained_teacher_dirname}"
		CUDA_VISIBLE_DEVICES=0 accelerate launch --mixed_precision fp16 train_student.py \
		--dataset $dataset \
		--model convnet6 \
		--teacher convnet8 \
		--teacher_path $trained_teacher_path \
		--teacher_loss focal_loss \
		--lr 0.1 \
		--epochs $epochs \
		--wd 1e-4 \
		--train-batch-size 128 \
		--checkpoint distilled_model_library \
		--T 5 --Lambda 0.95
	fi
fi

# Teacher: convnet10
if [[ "$student_model" == *"convnet6"* ]]; then
	## STUDENT=convnet6 ; TEACHER=convnet10_NLL
	if [[ "$teacher_model" == *"convnet10_cross_entropy"* ]]; then
		# The command below just outputs the checkpoint directory of the trained teacher model. In the next line full path to teacher model's checkpoint is created.
		trained_teacher_dirname=`bash find_checkpoint.sh convnet10_cross_entropy ${dataset}`
		trained_teacher_path="trained_model_library/${dataset}/convnet10/${trained_teacher_dirname}"
		CUDA_VISIBLE_DEVICES=0 accelerate launch --mixed_precision fp16 train_student.py \
		--dataset $dataset \
		--model convnet6 \
		--teacher convnet10 \
		--teacher_path $trained_teacher_path \
		--teacher_loss cross_entropy \
		--lr 0.1 \
		--epochs $epochs \
		--wd 1e-4 \
		--train-batch-size 128 \
		--checkpoint distilled_model_library \
		--T 5 --Lambda 0.95
	fi
fi

if [[ "$student_model" == *"convnet6"* ]]; then
	## STUDENT=convnet6 ; TEACHER=convnet10_NLL+MDCA
	if [[ "$teacher_model" == *"convnet10_NLL+MDCA"* ]]; then
		# The command below just outputs the checkpoint directory of the trained teacher model. In the next line full path to teacher model's checkpoint is created.
		trained_teacher_dirname=`bash find_checkpoint.sh convnet10_NLL+MDCA_beta=1.0 ${dataset}` # TODO: UPDATE THIS WITH THE BEST beta
		trained_teacher_path="trained_model_library/${dataset}/convnet10/${trained_teacher_dirname}"
		CUDA_VISIBLE_DEVICES=0 accelerate launch --mixed_precision fp16 train_student.py \
		--dataset $dataset \
		--model convnet6 \
		--teacher convnet10 \
		--teacher_path $trained_teacher_path \
		--teacher_loss NLL+MDCA \
		--lr 0.1 \
		--epochs $epochs \
		--wd 1e-4 \
		--train-batch-size 128 \
		--checkpoint distilled_model_library \
		--T 5 --Lambda 0.95
	fi
fi
 
if [[ "$student_model" == *"convnet6"* ]]; then
	## STUDENT=convnet6 ; TEACHER=convnet10_FL+MDCA
	if [[ "$teacher_model" == *"convnet10_FL+MDCA"* ]]; then
		# The command below just outputs the checkpoint directory of the trained teacher model. In the next line full path to teacher model's checkpoint is created.
		trained_teacher_dirname=`bash find_checkpoint.sh convnet10_FL+MDCA_gamma=1.0_beta=5.0 ${dataset}` # TODO: UPDATE THIS WITH THE BEST beta & gamma
		trained_teacher_path="trained_model_library/${dataset}/convnet10/${trained_teacher_dirname}"
		CUDA_VISIBLE_DEVICES=0 accelerate launch --mixed_precision fp16 train_student.py \
		--dataset $dataset \
		--model convnet6 \
		--teacher convnet10 \
		--teacher_path $trained_teacher_path \
		--teacher_loss FL+MDCA \
		--lr 0.1 \
		--epochs $epochs \
		--wd 1e-4 \
		--train-batch-size 128 \
		--checkpoint distilled_model_library \
		--T 5 --Lambda 0.95
	fi
fi

if [[ "$student_model" == *"convnet6"* ]]; then
	## STUDENT=convnet6 ; TEACHER=convnet10_focal_loss
	if [[ "$teacher_model" == *"convnet10_focal_loss"* ]]; then
		# The command below just outputs the checkpoint directory of the trained teacher model. In the next line full path to teacher model's checkpoint is created.
		trained_teacher_dirname=`bash find_checkpoint.sh convnet10_focal_loss_gamma=1.0 ${dataset}` # TODO: UPDATE THIS WITH THE BEST gamma
		trained_teacher_path="trained_model_library/${dataset}/convnet10/${trained_teacher_dirname}"
		CUDA_VISIBLE_DEVICES=0 accelerate launch --mixed_precision fp16 train_student.py \
		--dataset $dataset \
		--model convnet6 \
		--teacher convnet10 \
		--teacher_path $trained_teacher_path \
		--teacher_loss focal_loss \
		--lr 0.1 \
		--epochs $epochs \
		--wd 1e-4 \
		--train-batch-size 128 \
		--checkpoint distilled_model_library \
		--T 5 --Lambda 0.95
	fi
fi

########################################################################################################################################################
######################################################## convnet8 AS STUDENT ###########################################################################
########################################################################################################################################################

# Teacher: convnet2
if [[ "$student_model" == *"convnet8"* ]]; then
	## STUDENT=convnet8 ; TEACHER=convnet2_NLL
	if [[ "$teacher_model" == *"convnet2_cross_entropy"* ]]; then
		# The command below just outputs the checkpoint directory of the trained teacher model. In the next line full path to teacher model's checkpoint is created.
		trained_teacher_dirname=`bash find_checkpoint.sh convnet2_cross_entropy ${dataset}`
		trained_teacher_path="trained_model_library/${dataset}/convnet2/${trained_teacher_dirname}"
		CUDA_VISIBLE_DEVICES=0 accelerate launch --mixed_precision fp16 train_student.py \
		--dataset $dataset \
		--model convnet8 \
		--teacher convnet2 \
		--teacher_path $trained_teacher_path \
		--teacher_loss cross_entropy \
		--lr 0.1 \
		--epochs $epochs \
		--wd 1e-4 \
		--train-batch-size 128 \
		--checkpoint distilled_model_library \
		--T 5 --Lambda 0.95
	fi
fi

if [[ "$student_model" == *"convnet8"* ]]; then
	## STUDENT=convnet8 ; TEACHER=convnet2_NLL+MDCA
	if [[ "$teacher_model" == *"convnet2_NLL+MDCA"* ]]; then
		# The command below just outputs the checkpoint directory of the trained teacher model. In the next line full path to teacher model's checkpoint is created.
		trained_teacher_dirname=`bash find_checkpoint.sh convnet2_NLL+MDCA_beta=1.0 ${dataset}` # TODO: UPDATE THIS WITH THE BEST beta
		trained_teacher_path="trained_model_library/${dataset}/convnet2/${trained_teacher_dirname}"
		CUDA_VISIBLE_DEVICES=0 accelerate launch --mixed_precision fp16 train_student.py \
		--dataset $dataset \
		--model convnet8 \
		--teacher convnet2 \
		--teacher_path $trained_teacher_path \
		--teacher_loss NLL+MDCA \
		--lr 0.1 \
		--epochs $epochs \
		--wd 1e-4 \
		--train-batch-size 128 \
		--checkpoint distilled_model_library \
		--T 5 --Lambda 0.95
	fi
fi
 
if [[ "$student_model" == *"convnet8"* ]]; then
	## STUDENT=convnet8 ; TEACHER=convnet2_FL+MDCA
	if [[ "$teacher_model" == *"convnet2_FL+MDCA"* ]]; then
		# The command below just outputs the checkpoint directory of the trained teacher model. In the next line full path to teacher model's checkpoint is created.
		trained_teacher_dirname=`bash find_checkpoint.sh convnet2_FL+MDCA_gamma=1.0_beta=5.0 ${dataset}` # TODO: UPDATE THIS WITH THE BEST beta & gamma
		trained_teacher_path="trained_model_library/${dataset}/convnet2/${trained_teacher_dirname}"
		CUDA_VISIBLE_DEVICES=0 accelerate launch --mixed_precision fp16 train_student.py \
		--dataset $dataset \
		--model convnet8 \
		--teacher convnet2 \
		--teacher_path $trained_teacher_path \
		--teacher_loss FL+MDCA \
		--lr 0.1 \
		--epochs $epochs \
		--wd 1e-4 \
		--train-batch-size 128 \
		--checkpoint distilled_model_library \
		--T 5 --Lambda 0.95
	fi
fi

if [[ "$student_model" == *"convnet8"* ]]; then
	## STUDENT=convnet8 ; TEACHER=convnet2_focal_loss
	if [[ "$teacher_model" == *"convnet2_focal_loss"* ]]; then
		# The command below just outputs the checkpoint directory of the trained teacher model. In the next line full path to teacher model's checkpoint is created.
		trained_teacher_dirname=`bash find_checkpoint.sh convnet2_focal_loss_gamma=1.0 ${dataset}` # TODO: UPDATE THIS WITH THE BEST gamma
		trained_teacher_path="trained_model_library/${dataset}/convnet2/${trained_teacher_dirname}"
		CUDA_VISIBLE_DEVICES=0 accelerate launch --mixed_precision fp16 train_student.py \
		--dataset $dataset \
		--model convnet8 \
		--teacher convnet2 \
		--teacher_path $trained_teacher_path \
		--teacher_loss focal_loss \
		--lr 0.1 \
		--epochs $epochs \
		--wd 1e-4 \
		--train-batch-size 128 \
		--checkpoint distilled_model_library \
		--T 5 --Lambda 0.95
	fi
fi

# Teacher: convnet4
if [[ "$student_model" == *"convnet8"* ]]; then
	## STUDENT=convnet8 ; TEACHER=convnet4_NLL
	if [[ "$teacher_model" == *"convnet4_cross_entropy"* ]]; then
		# The command below just outputs the checkpoint directory of the trained teacher model. In the next line full path to teacher model's checkpoint is created.
		trained_teacher_dirname=`bash find_checkpoint.sh convnet4_cross_entropy ${dataset}`
		trained_teacher_path="trained_model_library/${dataset}/convnet4/${trained_teacher_dirname}"
		CUDA_VISIBLE_DEVICES=0 accelerate launch --mixed_precision fp16 train_student.py \
		--dataset $dataset \
		--model convnet8 \
		--teacher convnet4 \
		--teacher_path $trained_teacher_path \
		--teacher_loss cross_entropy \
		--lr 0.1 \
		--epochs $epochs \
		--wd 1e-4 \
		--train-batch-size 128 \
		--checkpoint distilled_model_library \
		--T 5 --Lambda 0.95
	fi
fi

if [[ "$student_model" == *"convnet8"* ]]; then
	## STUDENT=convnet8 ; TEACHER=convnet4_NLL+MDCA
	if [[ "$teacher_model" == *"convnet4_NLL+MDCA"* ]]; then
		# The command below just outputs the checkpoint directory of the trained teacher model. In the next line full path to teacher model's checkpoint is created.
		trained_teacher_dirname=`bash find_checkpoint.sh convnet4_NLL+MDCA_beta=1.0 ${dataset}` # TODO: UPDATE THIS WITH THE BEST beta
		trained_teacher_path="trained_model_library/${dataset}/convnet4/${trained_teacher_dirname}"
		CUDA_VISIBLE_DEVICES=0 accelerate launch --mixed_precision fp16 train_student.py \
		--dataset $dataset \
		--model convnet8 \
		--teacher convnet4 \
		--teacher_path $trained_teacher_path \
		--teacher_loss NLL+MDCA \
		--lr 0.1 \
		--epochs $epochs \
		--wd 1e-4 \
		--train-batch-size 128 \
		--checkpoint distilled_model_library \
		--T 5 --Lambda 0.95
	fi
fi
 
if [[ "$student_model" == *"convnet8"* ]]; then
	## STUDENT=convnet8 ; TEACHER=convnet4_FL+MDCA
	if [[ "$teacher_model" == *"convnet4_FL+MDCA"* ]]; then
		# The command below just outputs the checkpoint directory of the trained teacher model. In the next line full path to teacher model's checkpoint is created.
		trained_teacher_dirname=`bash find_checkpoint.sh convnet4_FL+MDCA_gamma=1.0_beta=5.0 ${dataset}` # TODO: UPDATE THIS WITH THE BEST beta & gamma
		trained_teacher_path="trained_model_library/${dataset}/convnet4/${trained_teacher_dirname}"
		CUDA_VISIBLE_DEVICES=0 accelerate launch --mixed_precision fp16 train_student.py \
		--dataset $dataset \
		--model convnet8 \
		--teacher convnet4 \
		--teacher_path $trained_teacher_path \
		--teacher_loss FL+MDCA \
		--lr 0.1 \
		--epochs $epochs \
		--wd 1e-4 \
		--train-batch-size 128 \
		--checkpoint distilled_model_library \
		--T 5 --Lambda 0.95
	fi
fi

if [[ "$student_model" == *"convnet8"* ]]; then
	## STUDENT=convnet8 ; TEACHER=convnet4_focal_loss
	if [[ "$teacher_model" == *"convnet4_focal_loss"* ]]; then
		# The command below just outputs the checkpoint directory of the trained teacher model. In the next line full path to teacher model's checkpoint is created.
		trained_teacher_dirname=`bash find_checkpoint.sh convnet4_focal_loss_gamma=1.0 ${dataset}` # TODO: UPDATE THIS WITH THE BEST gamma
		trained_teacher_path="trained_model_library/${dataset}/convnet4/${trained_teacher_dirname}"
		CUDA_VISIBLE_DEVICES=0 accelerate launch --mixed_precision fp16 train_student.py \
		--dataset $dataset \
		--model convnet8 \
		--teacher convnet4 \
		--teacher_path $trained_teacher_path \
		--teacher_loss focal_loss \
		--lr 0.1 \
		--epochs $epochs \
		--wd 1e-4 \
		--train-batch-size 128 \
		--checkpoint distilled_model_library \
		--T 5 --Lambda 0.95
	fi
fi

# Teacher: convnet6
if [[ "$student_model" == *"convnet8"* ]]; then
	## STUDENT=convnet8 ; TEACHER=convnet6_NLL
	if [[ "$teacher_model" == *"convnet6_cross_entropy"* ]]; then
		# The command below just outputs the checkpoint directory of the trained teacher model. In the next line full path to teacher model's checkpoint is created.
		trained_teacher_dirname=`bash find_checkpoint.sh convnet6_cross_entropy ${dataset}`
		trained_teacher_path="trained_model_library/${dataset}/convnet6/${trained_teacher_dirname}"
		CUDA_VISIBLE_DEVICES=0 accelerate launch --mixed_precision fp16 train_student.py \
		--dataset $dataset \
		--model convnet8 \
		--teacher convnet6 \
		--teacher_path $trained_teacher_path \
		--teacher_loss cross_entropy \
		--lr 0.1 \
		--epochs $epochs \
		--wd 1e-4 \
		--train-batch-size 128 \
		--checkpoint distilled_model_library \
		--T 5 --Lambda 0.95
	fi
fi

if [[ "$student_model" == *"convnet8"* ]]; then
	## STUDENT=convnet8 ; TEACHER=convnet6_NLL+MDCA
	if [[ "$teacher_model" == *"convnet6_NLL+MDCA"* ]]; then
		# The command below just outputs the checkpoint directory of the trained teacher model. In the next line full path to teacher model's checkpoint is created.
		trained_teacher_dirname=`bash find_checkpoint.sh convnet6_NLL+MDCA_beta=1.0 ${dataset}` # TODO: UPDATE THIS WITH THE BEST beta
		trained_teacher_path="trained_model_library/${dataset}/convnet6/${trained_teacher_dirname}"
		CUDA_VISIBLE_DEVICES=0 accelerate launch --mixed_precision fp16 train_student.py \
		--dataset $dataset \
		--model convnet8 \
		--teacher convnet6 \
		--teacher_path $trained_teacher_path \
		--teacher_loss NLL+MDCA \
		--lr 0.1 \
		--epochs $epochs \
		--wd 1e-4 \
		--train-batch-size 128 \
		--checkpoint distilled_model_library \
		--T 5 --Lambda 0.95
	fi
fi
 
if [[ "$student_model" == *"convnet8"* ]]; then
	## STUDENT=convnet8 ; TEACHER=convnet6_FL+MDCA
	if [[ "$teacher_model" == *"convnet6_FL+MDCA"* ]]; then
		# The command below just outputs the checkpoint directory of the trained teacher model. In the next line full path to teacher model's checkpoint is created.
		trained_teacher_dirname=`bash find_checkpoint.sh convnet6_FL+MDCA_gamma=1.0_beta=5.0 ${dataset}` # TODO: UPDATE THIS WITH THE BEST beta & gamma
		trained_teacher_path="trained_model_library/${dataset}/convnet6/${trained_teacher_dirname}"
		CUDA_VISIBLE_DEVICES=0 accelerate launch --mixed_precision fp16 train_student.py \
		--dataset $dataset \
		--model convnet8 \
		--teacher convnet6 \
		--teacher_path $trained_teacher_path \
		--teacher_loss FL+MDCA \
		--lr 0.1 \
		--epochs $epochs \
		--wd 1e-4 \
		--train-batch-size 128 \
		--checkpoint distilled_model_library \
		--T 5 --Lambda 0.95
	fi
fi

if [[ "$student_model" == *"convnet8"* ]]; then
	## STUDENT=convnet8 ; TEACHER=convnet6_focal_loss
	if [[ "$teacher_model" == *"convnet6_focal_loss"* ]]; then
		# The command below just outputs the checkpoint directory of the trained teacher model. In the next line full path to teacher model's checkpoint is created.
		trained_teacher_dirname=`bash find_checkpoint.sh convnet6_focal_loss_gamma=1.0 ${dataset}` # TODO: UPDATE THIS WITH THE BEST gamma
		trained_teacher_path="trained_model_library/${dataset}/convnet6/${trained_teacher_dirname}"
		CUDA_VISIBLE_DEVICES=0 accelerate launch --mixed_precision fp16 train_student.py \
		--dataset $dataset \
		--model convnet8 \
		--teacher convnet6 \
		--teacher_path $trained_teacher_path \
		--teacher_loss focal_loss \
		--lr 0.1 \
		--epochs $epochs \
		--wd 1e-4 \
		--train-batch-size 128 \
		--checkpoint distilled_model_library \
		--T 5 --Lambda 0.95
	fi
fi

# Teacher: convnet10
if [[ "$student_model" == *"convnet8"* ]]; then
	## STUDENT=convnet8 ; TEACHER=convnet10_NLL
	if [[ "$teacher_model" == *"convnet10_cross_entropy"* ]]; then
		# The command below just outputs the checkpoint directory of the trained teacher model. In the next line full path to teacher model's checkpoint is created.
		trained_teacher_dirname=`bash find_checkpoint.sh convnet10_cross_entropy ${dataset}`
		trained_teacher_path="trained_model_library/${dataset}/convnet10/${trained_teacher_dirname}"
		CUDA_VISIBLE_DEVICES=0 accelerate launch --mixed_precision fp16 train_student.py \
		--dataset $dataset \
		--model convnet8 \
		--teacher convnet10 \
		--teacher_path $trained_teacher_path \
		--teacher_loss cross_entropy \
		--lr 0.1 \
		--epochs $epochs \
		--wd 1e-4 \
		--train-batch-size 128 \
		--checkpoint distilled_model_library \
		--T 5 --Lambda 0.95
	fi
fi

if [[ "$student_model" == *"convnet8"* ]]; then
	## STUDENT=convnet8 ; TEACHER=convnet10_NLL+MDCA
	if [[ "$teacher_model" == *"convnet10_NLL+MDCA"* ]]; then
		# The command below just outputs the checkpoint directory of the trained teacher model. In the next line full path to teacher model's checkpoint is created.
		trained_teacher_dirname=`bash find_checkpoint.sh convnet10_NLL+MDCA_beta=1.0 ${dataset}` # TODO: UPDATE THIS WITH THE BEST beta
		trained_teacher_path="trained_model_library/${dataset}/convnet10/${trained_teacher_dirname}"
		CUDA_VISIBLE_DEVICES=0 accelerate launch --mixed_precision fp16 train_student.py \
		--dataset $dataset \
		--model convnet8 \
		--teacher convnet10 \
		--teacher_path $trained_teacher_path \
		--teacher_loss NLL+MDCA \
		--lr 0.1 \
		--epochs $epochs \
		--wd 1e-4 \
		--train-batch-size 128 \
		--checkpoint distilled_model_library \
		--T 5 --Lambda 0.95
	fi
fi
 
if [[ "$student_model" == *"convnet8"* ]]; then
	## STUDENT=convnet8 ; TEACHER=convnet10_FL+MDCA
	if [[ "$teacher_model" == *"convnet10_FL+MDCA"* ]]; then
		# The command below just outputs the checkpoint directory of the trained teacher model. In the next line full path to teacher model's checkpoint is created.
		trained_teacher_dirname=`bash find_checkpoint.sh convnet10_FL+MDCA_gamma=1.0_beta=5.0 ${dataset}` # TODO: UPDATE THIS WITH THE BEST beta & gamma
		trained_teacher_path="trained_model_library/${dataset}/convnet10/${trained_teacher_dirname}"
		CUDA_VISIBLE_DEVICES=0 accelerate launch --mixed_precision fp16 train_student.py \
		--dataset $dataset \
		--model convnet8 \
		--teacher convnet10 \
		--teacher_path $trained_teacher_path \
		--teacher_loss FL+MDCA \
		--lr 0.1 \
		--epochs $epochs \
		--wd 1e-4 \
		--train-batch-size 128 \
		--checkpoint distilled_model_library \
		--T 5 --Lambda 0.95
	fi
fi

if [[ "$student_model" == *"convnet8"* ]]; then
	## STUDENT=convnet8 ; TEACHER=convnet10_focal_loss
	if [[ "$teacher_model" == *"convnet10_focal_loss"* ]]; then
		# The command below just outputs the checkpoint directory of the trained teacher model. In the next line full path to teacher model's checkpoint is created.
		trained_teacher_dirname=`bash find_checkpoint.sh convnet10_focal_loss_gamma=1.0 ${dataset}` # TODO: UPDATE THIS WITH THE BEST gamma
		trained_teacher_path="trained_model_library/${dataset}/convnet10/${trained_teacher_dirname}"
		CUDA_VISIBLE_DEVICES=0 accelerate launch --mixed_precision fp16 train_student.py \
		--dataset $dataset \
		--model convnet8 \
		--teacher convnet10 \
		--teacher_path $trained_teacher_path \
		--teacher_loss focal_loss \
		--lr 0.1 \
		--epochs $epochs \
		--wd 1e-4 \
		--train-batch-size 128 \
		--checkpoint distilled_model_library \
		--T 5 --Lambda 0.95
	fi
fi

########################################################################################################################################################
######################################################## convnet10 AS STUDENT ###########################################################################
########################################################################################################################################################

# Teacher: convnet2
if [[ "$student_model" == *"convnet10"* ]]; then
	## STUDENT=convnet10 ; TEACHER=convnet2_NLL
	if [[ "$teacher_model" == *"convnet2_cross_entropy"* ]]; then
		# The command below just outputs the checkpoint directory of the trained teacher model. In the next line full path to teacher model's checkpoint is created.
		trained_teacher_dirname=`bash find_checkpoint.sh convnet2_cross_entropy ${dataset}`
		trained_teacher_path="trained_model_library/${dataset}/convnet2/${trained_teacher_dirname}"
		CUDA_VISIBLE_DEVICES=0 accelerate launch --mixed_precision fp16 train_student.py \
		--dataset $dataset \
		--model convnet10 \
		--teacher convnet2 \
		--teacher_path $trained_teacher_path \
		--teacher_loss cross_entropy \
		--lr 0.1 \
		--epochs $epochs \
		--wd 1e-4 \
		--train-batch-size 128 \
		--checkpoint distilled_model_library \
		--T 5 --Lambda 0.95
	fi
fi

if [[ "$student_model" == *"convnet10"* ]]; then
	## STUDENT=convnet10 ; TEACHER=convnet2_NLL+MDCA
	if [[ "$teacher_model" == *"convnet2_NLL+MDCA"* ]]; then
		# The command below just outputs the checkpoint directory of the trained teacher model. In the next line full path to teacher model's checkpoint is created.
		trained_teacher_dirname=`bash find_checkpoint.sh convnet2_NLL+MDCA_beta=1.0 ${dataset}` # TODO: UPDATE THIS WITH THE BEST beta
		trained_teacher_path="trained_model_library/${dataset}/convnet2/${trained_teacher_dirname}"
		CUDA_VISIBLE_DEVICES=0 accelerate launch --mixed_precision fp16 train_student.py \
		--dataset $dataset \
		--model convnet10 \
		--teacher convnet2 \
		--teacher_path $trained_teacher_path \
		--teacher_loss NLL+MDCA \
		--lr 0.1 \
		--epochs $epochs \
		--wd 1e-4 \
		--train-batch-size 128 \
		--checkpoint distilled_model_library \
		--T 5 --Lambda 0.95
	fi
fi
 
if [[ "$student_model" == *"convnet10"* ]]; then
	## STUDENT=convnet10 ; TEACHER=convnet2_FL+MDCA
	if [[ "$teacher_model" == *"convnet2_FL+MDCA"* ]]; then
		# The command below just outputs the checkpoint directory of the trained teacher model. In the next line full path to teacher model's checkpoint is created.
		trained_teacher_dirname=`bash find_checkpoint.sh convnet2_FL+MDCA_gamma=1.0_beta=5.0 ${dataset}` # TODO: UPDATE THIS WITH THE BEST beta & gamma
		trained_teacher_path="trained_model_library/${dataset}/convnet2/${trained_teacher_dirname}"
		CUDA_VISIBLE_DEVICES=0 accelerate launch --mixed_precision fp16 train_student.py \
		--dataset $dataset \
		--model convnet10 \
		--teacher convnet2 \
		--teacher_path $trained_teacher_path \
		--teacher_loss FL+MDCA \
		--lr 0.1 \
		--epochs $epochs \
		--wd 1e-4 \
		--train-batch-size 128 \
		--checkpoint distilled_model_library \
		--T 5 --Lambda 0.95
	fi
fi

if [[ "$student_model" == *"convnet10"* ]]; then
	## STUDENT=convnet10 ; TEACHER=convnet2_focal_loss
	if [[ "$teacher_model" == *"convnet2_focal_loss"* ]]; then
		# The command below just outputs the checkpoint directory of the trained teacher model. In the next line full path to teacher model's checkpoint is created.
		trained_teacher_dirname=`bash find_checkpoint.sh convnet2_focal_loss_gamma=1.0 ${dataset}` # TODO: UPDATE THIS WITH THE BEST gamma
		trained_teacher_path="trained_model_library/${dataset}/convnet2/${trained_teacher_dirname}"
		CUDA_VISIBLE_DEVICES=0 accelerate launch --mixed_precision fp16 train_student.py \
		--dataset $dataset \
		--model convnet10 \
		--teacher convnet2 \
		--teacher_path $trained_teacher_path \
		--teacher_loss focal_loss \
		--lr 0.1 \
		--epochs $epochs \
		--wd 1e-4 \
		--train-batch-size 128 \
		--checkpoint distilled_model_library \
		--T 5 --Lambda 0.95
	fi
fi

# Teacher: convnet4
if [[ "$student_model" == *"convnet10"* ]]; then
	## STUDENT=convnet10 ; TEACHER=convnet4_NLL
	if [[ "$teacher_model" == *"convnet4_cross_entropy"* ]]; then
		# The command below just outputs the checkpoint directory of the trained teacher model. In the next line full path to teacher model's checkpoint is created.
		trained_teacher_dirname=`bash find_checkpoint.sh convnet4_cross_entropy ${dataset}`
		trained_teacher_path="trained_model_library/${dataset}/convnet4/${trained_teacher_dirname}"
		CUDA_VISIBLE_DEVICES=0 accelerate launch --mixed_precision fp16 train_student.py \
		--dataset $dataset \
		--model convnet10 \
		--teacher convnet4 \
		--teacher_path $trained_teacher_path \
		--teacher_loss cross_entropy \
		--lr 0.1 \
		--epochs $epochs \
		--wd 1e-4 \
		--train-batch-size 128 \
		--checkpoint distilled_model_library \
		--T 5 --Lambda 0.95
	fi
fi

if [[ "$student_model" == *"convnet10"* ]]; then
	## STUDENT=convnet10 ; TEACHER=convnet4_NLL+MDCA
	if [[ "$teacher_model" == *"convnet4_NLL+MDCA"* ]]; then
		# The command below just outputs the checkpoint directory of the trained teacher model. In the next line full path to teacher model's checkpoint is created.
		trained_teacher_dirname=`bash find_checkpoint.sh convnet4_NLL+MDCA_beta=1.0 ${dataset}` # TODO: UPDATE THIS WITH THE BEST beta
		trained_teacher_path="trained_model_library/${dataset}/convnet4/${trained_teacher_dirname}"
		CUDA_VISIBLE_DEVICES=0 accelerate launch --mixed_precision fp16 train_student.py \
		--dataset $dataset \
		--model convnet10 \
		--teacher convnet4 \
		--teacher_path $trained_teacher_path \
		--teacher_loss NLL+MDCA \
		--lr 0.1 \
		--epochs $epochs \
		--wd 1e-4 \
		--train-batch-size 128 \
		--checkpoint distilled_model_library \
		--T 5 --Lambda 0.95
	fi
fi
 
if [[ "$student_model" == *"convnet10"* ]]; then
	## STUDENT=convnet10 ; TEACHER=convnet4_FL+MDCA
	if [[ "$teacher_model" == *"convnet4_FL+MDCA"* ]]; then
		# The command below just outputs the checkpoint directory of the trained teacher model. In the next line full path to teacher model's checkpoint is created.
		trained_teacher_dirname=`bash find_checkpoint.sh convnet4_FL+MDCA_gamma=1.0_beta=5.0 ${dataset}` # TODO: UPDATE THIS WITH THE BEST beta & gamma
		trained_teacher_path="trained_model_library/${dataset}/convnet4/${trained_teacher_dirname}"
		CUDA_VISIBLE_DEVICES=0 accelerate launch --mixed_precision fp16 train_student.py \
		--dataset $dataset \
		--model convnet10 \
		--teacher convnet4 \
		--teacher_path $trained_teacher_path \
		--teacher_loss FL+MDCA \
		--lr 0.1 \
		--epochs $epochs \
		--wd 1e-4 \
		--train-batch-size 128 \
		--checkpoint distilled_model_library \
		--T 5 --Lambda 0.95
	fi
fi

if [[ "$student_model" == *"convnet10"* ]]; then
	## STUDENT=convnet10 ; TEACHER=convnet4_focal_loss
	if [[ "$teacher_model" == *"convnet4_focal_loss"* ]]; then
		# The command below just outputs the checkpoint directory of the trained teacher model. In the next line full path to teacher model's checkpoint is created.
		trained_teacher_dirname=`bash find_checkpoint.sh convnet4_focal_loss_gamma=1.0 ${dataset}` # TODO: UPDATE THIS WITH THE BEST gamma
		trained_teacher_path="trained_model_library/${dataset}/convnet4/${trained_teacher_dirname}"
		CUDA_VISIBLE_DEVICES=0 accelerate launch --mixed_precision fp16 train_student.py \
		--dataset $dataset \
		--model convnet10 \
		--teacher convnet4 \
		--teacher_path $trained_teacher_path \
		--teacher_loss focal_loss \
		--lr 0.1 \
		--epochs $epochs \
		--wd 1e-4 \
		--train-batch-size 128 \
		--checkpoint distilled_model_library \
		--T 5 --Lambda 0.95
	fi
fi

# Teacher: convnet6
if [[ "$student_model" == *"convnet10"* ]]; then
	## STUDENT=convnet10 ; TEACHER=convnet6_NLL
	if [[ "$teacher_model" == *"convnet6_cross_entropy"* ]]; then
		# The command below just outputs the checkpoint directory of the trained teacher model. In the next line full path to teacher model's checkpoint is created.
		trained_teacher_dirname=`bash find_checkpoint.sh convnet6_cross_entropy ${dataset}`
		trained_teacher_path="trained_model_library/${dataset}/convnet6/${trained_teacher_dirname}"
		CUDA_VISIBLE_DEVICES=0 accelerate launch --mixed_precision fp16 train_student.py \
		--dataset $dataset \
		--model convnet10 \
		--teacher convnet6 \
		--teacher_path $trained_teacher_path \
		--teacher_loss cross_entropy \
		--lr 0.1 \
		--epochs $epochs \
		--wd 1e-4 \
		--train-batch-size 128 \
		--checkpoint distilled_model_library \
		--T 5 --Lambda 0.95
	fi
fi

if [[ "$student_model" == *"convnet10"* ]]; then
	## STUDENT=convnet10 ; TEACHER=convnet6_NLL+MDCA
	if [[ "$teacher_model" == *"convnet6_NLL+MDCA"* ]]; then
		# The command below just outputs the checkpoint directory of the trained teacher model. In the next line full path to teacher model's checkpoint is created.
		trained_teacher_dirname=`bash find_checkpoint.sh convnet6_NLL+MDCA_beta=1.0 ${dataset}` # TODO: UPDATE THIS WITH THE BEST beta
		trained_teacher_path="trained_model_library/${dataset}/convnet6/${trained_teacher_dirname}"
		CUDA_VISIBLE_DEVICES=0 accelerate launch --mixed_precision fp16 train_student.py \
		--dataset $dataset \
		--model convnet10 \
		--teacher convnet6 \
		--teacher_path $trained_teacher_path \
		--teacher_loss NLL+MDCA \
		--lr 0.1 \
		--epochs $epochs \
		--wd 1e-4 \
		--train-batch-size 128 \
		--checkpoint distilled_model_library \
		--T 5 --Lambda 0.95
	fi
fi
 
if [[ "$student_model" == *"convnet10"* ]]; then
	## STUDENT=convnet10 ; TEACHER=convnet6_FL+MDCA
	if [[ "$teacher_model" == *"convnet6_FL+MDCA"* ]]; then
		# The command below just outputs the checkpoint directory of the trained teacher model. In the next line full path to teacher model's checkpoint is created.
		trained_teacher_dirname=`bash find_checkpoint.sh convnet6_FL+MDCA_gamma=1.0_beta=5.0 ${dataset}` # TODO: UPDATE THIS WITH THE BEST beta & gamma
		trained_teacher_path="trained_model_library/${dataset}/convnet6/${trained_teacher_dirname}"
		CUDA_VISIBLE_DEVICES=0 accelerate launch --mixed_precision fp16 train_student.py \
		--dataset $dataset \
		--model convnet10 \
		--teacher convnet6 \
		--teacher_path $trained_teacher_path \
		--teacher_loss FL+MDCA \
		--lr 0.1 \
		--epochs $epochs \
		--wd 1e-4 \
		--train-batch-size 128 \
		--checkpoint distilled_model_library \
		--T 5 --Lambda 0.95
	fi
fi

if [[ "$student_model" == *"convnet10"* ]]; then
	## STUDENT=convnet10 ; TEACHER=convnet6_focal_loss
	if [[ "$teacher_model" == *"convnet6_focal_loss"* ]]; then
		# The command below just outputs the checkpoint directory of the trained teacher model. In the next line full path to teacher model's checkpoint is created.
		trained_teacher_dirname=`bash find_checkpoint.sh convnet6_focal_loss_gamma=1.0 ${dataset}` # TODO: UPDATE THIS WITH THE BEST gamma
		trained_teacher_path="trained_model_library/${dataset}/convnet6/${trained_teacher_dirname}"
		CUDA_VISIBLE_DEVICES=0 accelerate launch --mixed_precision fp16 train_student.py \
		--dataset $dataset \
		--model convnet10 \
		--teacher convnet6 \
		--teacher_path $trained_teacher_path \
		--teacher_loss focal_loss \
		--lr 0.1 \
		--epochs $epochs \
		--wd 1e-4 \
		--train-batch-size 128 \
		--checkpoint distilled_model_library \
		--T 5 --Lambda 0.95
	fi
fi

# Teacher: convnet8
if [[ "$student_model" == *"convnet10"* ]]; then
	## STUDENT=convnet10 ; TEACHER=convnet8_NLL
	if [[ "$teacher_model" == *"convnet8_cross_entropy"* ]]; then
		# The command below just outputs the checkpoint directory of the trained teacher model. In the next line full path to teacher model's checkpoint is created.
		trained_teacher_dirname=`bash find_checkpoint.sh convnet8_cross_entropy ${dataset}`
		trained_teacher_path="trained_model_library/${dataset}/convnet8/${trained_teacher_dirname}"
		CUDA_VISIBLE_DEVICES=0 accelerate launch --mixed_precision fp16 train_student.py \
		--dataset $dataset \
		--model convnet10 \
		--teacher convnet8 \
		--teacher_path $trained_teacher_path \
		--teacher_loss cross_entropy \
		--lr 0.1 \
		--epochs $epochs \
		--wd 1e-4 \
		--train-batch-size 128 \
		--checkpoint distilled_model_library \
		--T 5 --Lambda 0.95
	fi
fi

if [[ "$student_model" == *"convnet10"* ]]; then
	## STUDENT=convnet10 ; TEACHER=convnet8_NLL+MDCA
	if [[ "$teacher_model" == *"convnet8_NLL+MDCA"* ]]; then
		# The command below just outputs the checkpoint directory of the trained teacher model. In the next line full path to teacher model's checkpoint is created.
		trained_teacher_dirname=`bash find_checkpoint.sh convnet8_NLL+MDCA_beta=1.0 ${dataset}` # TODO: UPDATE THIS WITH THE BEST beta
		trained_teacher_path="trained_model_library/${dataset}/convnet8/${trained_teacher_dirname}"
		CUDA_VISIBLE_DEVICES=0 accelerate launch --mixed_precision fp16 train_student.py \
		--dataset $dataset \
		--model convnet10 \
		--teacher convnet8 \
		--teacher_path $trained_teacher_path \
		--teacher_loss NLL+MDCA \
		--lr 0.1 \
		--epochs $epochs \
		--wd 1e-4 \
		--train-batch-size 128 \
		--checkpoint distilled_model_library \
		--T 5 --Lambda 0.95
	fi
fi
 
if [[ "$student_model" == *"convnet10"* ]]; then
	## STUDENT=convnet10 ; TEACHER=convnet8_FL+MDCA
	if [[ "$teacher_model" == *"convnet8_FL+MDCA"* ]]; then
		# The command below just outputs the checkpoint directory of the trained teacher model. In the next line full path to teacher model's checkpoint is created.
		trained_teacher_dirname=`bash find_checkpoint.sh convnet8_FL+MDCA_gamma=1.0_beta=5.0 ${dataset}` # TODO: UPDATE THIS WITH THE BEST beta & gamma
		trained_teacher_path="trained_model_library/${dataset}/convnet8/${trained_teacher_dirname}"
		CUDA_VISIBLE_DEVICES=0 accelerate launch --mixed_precision fp16 train_student.py \
		--dataset $dataset \
		--model convnet10 \
		--teacher convnet8 \
		--teacher_path $trained_teacher_path \
		--teacher_loss FL+MDCA \
		--lr 0.1 \
		--epochs $epochs \
		--wd 1e-4 \
		--train-batch-size 128 \
		--checkpoint distilled_model_library \
		--T 5 --Lambda 0.95
	fi
fi

if [[ "$student_model" == *"convnet10"* ]]; then
	## STUDENT=convnet10 ; TEACHER=convnet8_focal_loss
	if [[ "$teacher_model" == *"convnet8_focal_loss"* ]]; then
		# The command below just outputs the checkpoint directory of the trained teacher model. In the next line full path to teacher model's checkpoint is created.
		trained_teacher_dirname=`bash find_checkpoint.sh convnet8_focal_loss_gamma=1.0 ${dataset}` # TODO: UPDATE THIS WITH THE BEST gamma
		trained_teacher_path="trained_model_library/${dataset}/convnet8/${trained_teacher_dirname}"
		CUDA_VISIBLE_DEVICES=0 accelerate launch --mixed_precision fp16 train_student.py \
		--dataset $dataset \
		--model convnet10 \
		--teacher convnet8 \
		--teacher_path $trained_teacher_path \
		--teacher_loss focal_loss \
		--lr 0.1 \
		--epochs $epochs \
		--wd 1e-4 \
		--train-batch-size 128 \
		--checkpoint distilled_model_library \
		--T 5 --Lambda 0.95
	fi
fi

