#!/bin/sh
### General options
### –- specify queue --
#BSUB -q gpuqueuek80
### -- set the job Name --
#BSUB -J autoencoder
### -- ask for number of cores (default: 1) --
###BSUB -n 1
### -- set walltime limit: hh:mm --
#BSUB -W 6:00
### -- set the email address --
# please uncomment the following line and put in your e-mail address,
# if you want to receive e-mail notifications on a non-default address
#BSUB -u arydbirk@gmail.com
### -- send notification at start --
#BSUB -B
### -- Select the resources: 2 gpus in exclusive process mode --
###BSUB -R "rusage[ngpus_excl_p=1]"
#BSUB -R "rusage[ngpus_excl_p=1]"
### -- send notification at completion --
#BSUB -N
### -- Specify the output and error file. %J is the job-id --
### -- -o and -e mean append, -oo and -eo mean overwrite --
#BSUB -o o%J.out
#BSUB -e e%J.err

module load cudnn/v6.0-prod
module load python3/3.5.1
source /appl/tensorflow/1.3gpu-python3.5/bin/activate

/appl/glibc/2.17/lib/ld-linux-x86-64.so.2  --library-path /appl/glibc/2.17/lib/:/appl/gcc/4.8.5/lib64/:/usr/lib64/atlas:/lib64:/usr/lib64:$LD_LIBRARY_PATH $(which python) main_encoder.py


