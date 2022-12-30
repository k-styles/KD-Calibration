#!bin/bash/
# Author: Kartik Anand
# Incase of any bug or inconsistency or anything related to this file, contact me.

# Below Note Fixed now. Although it's better to have this convention.
#######################################################################################################################
# NOTE: Make sure you have used gamma before beta in the string argument. Example: resnet56_FL+MDCA_gamma=1.0_beta=3.0#
#######################################################################################################################

###################### YOU ARE NOT RECOMMENDED TO LOOK AT THIS FILE IF YOU'RE TRYING TO UNDERSTAND THE CODE AS A WHOLE ###########################
##### JUST GET THIS: This script takes a description of a teacher model you want to use from your trained_library_checkpoint/ directory and automatically gets from there by reading out the proper model directory names in there, and outputs the exact checkpoint you want to use. THIS SCRIPT WAS PRIMARILY CREATED FOR KD_script.sh

### EXAMPLE:
# > bash find_checkpoint.sh resnet56_FL+MDCA_gamma=2.0_beta=10.0 cifar10
###
model_description=$1
dataset=$2

model=`echo $1 | cut -f1 -d_`
loss=`echo $1 | cut -f2 -d_`
if [[ "$loss" == "cross" ]]; then
	loss="cross_entropy"
fi
if [[ "$loss" == "focal" ]]; then
	loss="focal_loss"
	gamma=`echo $1 | cut -f4 -d_ | cut -f2 -d=`
fi
if [[ "$loss" == "NLL+MDCA" ]]; then
	beta=`echo $1 | cut -f3 -d_ | cut -f2 -d=`
fi
if [[ "$loss" == "FL+MDCA" ]]; then
	#echo "[find_checkpoints.sh]: NOTE: Make sure you have used gamma before beta in the string argument. Example: resnet56_FL+MDCA_gamma=1.0_beta=3.0"
	hyperparam1=`echo $1 | cut -f3 -d_`
	hyperparam2=`echo $1 | cut -f4 -d_`
	if [[ "$hyperparam1" == *"gamma"* ]]; then
		gamma=`echo $1 | cut -f3 -d_ | cut -f2 -d=`
		beta=`echo $1 | cut -f4 -d_ | cut -f2 -d=`
	elif [[ "$hyperparam1" == *"beta"* ]]; then
		beta=`echo $1 | cut -f3 -d_ | cut -f2 -d=`
		gamma=`echo $1 | cut -f4 -d_ | cut -f2 -d=`
		hyperparam1="gamma=$gamma"
		hyperparam2="beta=$beta"
	else
		echo "[ERROR]: Syntactical error in first argument: $1"
		exit 125
	fi
		
		
		
fi

echo " " > dummy
for FILE in trained_model_library/$dataset/$model/*;
do
	echo $FILE >> dummy;
done

# This loop actually loops through the model directories' names
# dummy file is read backwards because if multiple model directories with same loss and same hyperparameters is found, most recent model directory is given the preference
tac dummy | head -n -1 | while read line;
do
	if [[ $loss == "cross_entropy" ]]; then
		if [[ $line == *"cross_entropy"* ]]; then
			checkpoint=`echo $line | cut -f4 -d/`
			echo $checkpoint
			exit 0
		fi
	fi
	if [[ $loss == "focal_loss" ]]; then
		if [[ $line == *"focal_loss_gamma=$gamma"* ]]; then
			checkpoint=`echo $line | cut -f4 -d/`
			echo $checkpoint
			exit 0
		fi
	fi
	if [[ $loss == "FL+MDCA" ]]; then
		if [[ $line == *"FL+MDCA_${hyperparam1}_${hyperparam2}"* ]]; then
			checkpoint=`echo $line | cut -f4 -d/`
			echo $checkpoint
			exit 0
		fi
	fi
	if [[ $loss == "NLL+MDCA" ]]; then
		if [[ $line == *"NLL+MDCA_beta=$beta"* ]]; then
			checkpoint=`echo $line | cut -f4 -d/`
			echo $checkpoint
			exit 0
		fi
	fi
done
rm -rf dummy
