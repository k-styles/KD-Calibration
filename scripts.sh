### CIFAR10/100 ###############################################################

# train teacher resnet56 on cross_entropy ####################################
CUDA_VISIBLE_DEVICES=0 accelerate launch --mixed_precision fp16 train_teacher.py \
--dataset cifar100 \
--model resnet56 \
--lr 0.1 \
--epochs 160 \
--scheduler multistep \
--schedule-steps 80 120 \
--lr-decay-factor 0.1 \
--wd 1e-4 \
--train-batch-size 128 \
--loss cross_entropy

# train teacher resnet56 on FL+MDCA ###########################
CUDA_VISIBLE_DEVICES=0 accelerate launch --mixed_precision fp16 train_teacher.py \
--dataset cifar100 \
--model resnet56 \
--lr 0.1 \
--epochs 160 \
--scheduler multistep \
--schedule-steps 80 120 \
--lr-decay-factor 0.1 \
--wd 1e-4 \
--train-batch-size 128 \
--loss FL+MDCA --gamma 3.0 --beta 1.0

# train teacher convnet on cross entropy ###########################
CUDA_VISIBLE_DEVICES=0 accelerate launch --mixed_precision fp16 train_teacher.py \
--dataset cifar100 \
--model convnet10 \
--lr 0.1 \
--epochs 160 \
--wd 1e-4 \
--train-batch-size 128 \
--loss cross_entropy

# Train students #############################################
CUDA_VISIBLE_DEVICES=0 accelerate launch --mixed_precision fp16 train_student.py \
--dataset cifar100 \
--model resnet8 \
--teacher resnet56 \
--teacher_path checkpoint/cifar100/2022-10-06-17:52:57.291536_resnet56_cross_entropy \
--lr 0.1 \
--epochs 160 \
--wd 1e-4 \
--train-batch-size 128 \
--T 5 --Lambda 0.95

###############################################################################################################

# train resnet on tiny_imagenet using warmup_cosine lr scheduling
# currently 352 steps per epoch on 4 GPUs using 64 batch-size per GPU
CUDA_VISIBLE_DEVICES=0,1,2,3 accelerate launch --mixed_precision fp16 train_teacher.py \
--dataset tiny_imagenet \
--model resnet18_tin \
--lr 0.1 \
--scheduler warmupcosine \
--warmup 1000 \
--wd 1e-4 \
--train-batch-size 64 \
--epochs 50 \
--loss cross_entropy

# train a student resnet on tiny_imagenet
# same config as above
CUDA_VISIBLE_DEVICES=0,1,2,3 accelerate launch --mixed_precision fp16 train_student.py \
--dataset tiny_imagenet \
--model resnet18_tin \
--lr 0.1 \
--scheduler warmupcosine \
--warmup 1000 \
--wd 1e-4 \
--train-batch-size 64 \
--epochs 5 \
--dw 1.0 \
--temp 1.0 \
--teacher resnet152_tin --checkpoint checkpoint/tiny_imagenet/2022-09-29-17:03:39.337810_resnet152_tin_cross_entropy/model_best.pth


# train a student using OE trained model
CUDA_VISIBLE_DEVICES=0,1,2,3 accelerate launch --mixed_precision fp16 train_student.py \
--dataset cifar100 \
--model wide-resnet-40-1 \
--lr 0.1 \
--scheduler warmupcosine \
--wd 5e-4 \
--train-batch-size 64 \
--epochs 100 \
--dw 1.0 \
--temp 1.0 \
--teacher wide-resnet-40-2 \
--checkpoint pretrained_models/wide-resnet-40-2_cifar100/cifar100_wrn_oe_tune_epoch_9.pt

# train convnet
python train_teacher.py \
--dataset cifar10 \
--model convnet \
--lr 0.001 \
--wd 1e-4 \
--train-batch-size 128 \
--epochs 200 \
--loss cross_entropy


# try to run on all configurations for (gamma, beta) pairs gamma in [1,2,3] beta in [1,5,10]
CUDA_VISIBLE_DEVICES=7 python train_teacher.py \
--dataset cifar100 \
--model resnet110 \
--lr 0.1 \
--lr-decay-factor 0.1 \
--wd 5e-4 \
--train-batch-size 128 \
--schedule-steps 100 150 \
--epochs 200 \
--loss FL+MDCA \
--gamma 1.0 \
--beta 1.0

# try for all gamma values = [1, 2, 3]
CUDA_VISIBLE_DEVICES=7 python train_teacher.py \
--dataset cifar100 \
--model resnet110 \
--lr 0.1 \
--lr-decay-factor 0.1 \
--wd 5e-4 \
--train-batch-size 128 \
--schedule-steps 100 150 \
--epochs 200 \
--loss focal_loss \
--gamma 1.0



# Free gpus = [5, 6, 7]
# run the following commmand
simple_gpu_scheduler --gpus 5 6 7 < gpu_commands_hard.txt


# not so free  = [0, 1, 2, 3, 4, 5, 6, 7]
# run the following commmand
simple_gpu_scheduler --gpus 0 1 2 3 4 5 6 7 < gpu_commands_easy.txt

# Train cifar10
# Just replace resnet20 with other model names such as resnet18, resnet50, resnet110 to train on them
# you can also tweak hyper-parameters, look at utils/argparser.py for more parameters.

# teacher training on CIFAR10
CUDA_VISIBLE_DEVICES=6 python train_teacher.py \
--dataset cifar100 \
--model resnet50 \
--lr 0.1 \
--lr-decay-factor 0.1 \
--wd 5e-4 \
--train-batch-size 128 \
--schedule-steps 100 150 \
--epochs 200 \
--loss mdca \
--gamma 3.0 \
--beta 10.0

CUDA_VISIBLE_DEVICES=6 python train_teacher.py \
--dataset cifar100 \
--model resnet50 \
--lr 0.1 \
--lr-decay-factor 0.1 \
--wd 5e-4 \
--train-batch-size 128 \
--schedule-steps 100 150 \
--epochs 200 \
--loss cross_entropy

# loss = [cross_entropy, mdca]

CUDA_VISIBLE_DEVICES=7 python train_student.py \
--dataset cifar10 \
--model resnet18 \
--teacher resnet152 \
--checkpoint checkpoint/cifar10/15-May_resnet152_cross_entropy/model_best.pth \
--lr 0.1 \
--lr-decay-factor 0.1 \
--wd 5e-4 \
--train-batch-size 128 \
--schedule-steps 100 150 \
--epochs 200
