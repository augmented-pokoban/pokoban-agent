#!/bin/sh
### General options
### –- specify queue --
#BSUB -q gputitanxpascal
### -- set the job Name --
#BSUB -J uns_gpu
### -- ask for number of cores (default: 1) --
###BSUB -n 4
### -- set walltime limit: hh:mm --
#BSUB -W 12:00
### -- Select the resources: 1 gpu in exclusive process mode --
#BSUB -gpu "num=4:mode=exclusive_process"
# request memory
#BSUB -R "rusage[mem=8GB]"
### -- set the email address --
# please uncomment the following line and put in your e-mail address,
# if you want to receive e-mail notifications on a non-default address
#BSUB -u arydbirk@gmail.com
### -- send notification at start --
#BSUB -B
### -- send notification at completion --
#BSUB -N
### -- Specify the output and error file. %J is the job-id --
### -- -o and -e mean append, -oo and -eo mean overwrite --
#BSUB -o o%J.out
#BSUB -e e%J.err

module load cudnn/v6.0-prod
module load python3/3.6.2
source /appl/tensorflow/1.4gpu-python362/bin/activate
python3 main_policy.py

