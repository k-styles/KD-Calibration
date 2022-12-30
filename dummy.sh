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

