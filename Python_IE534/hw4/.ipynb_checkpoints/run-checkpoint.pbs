#!/bin/bash
#PBS -l nodes=1:ppn=16:xk
#PBS -N hw4_tf
#PBS -l walltime=02:00:00
#PBS -e $PBS_JOBNAME.$PBS_JOBID.err
#PBS -o $PBS_JOBNAME.$PBS_JOBID.out
#PBS -M yutongd3@illinois.edu
cd /u/training/tra380/IE534/hw4
. /opt/modules/default/init/bash
module load bwpy
module load cudatoolkit
aprun -n 1 -N 1 python3.6 hw4_resnet.py --num_epochs 50 --batch_size 500 --resume './myresnet_checkpoint.pth.tar'
aprun -n 1 -N 1 python3.6 hw4_transfer_learning.py --num_epochs 30 --batch_size 100 --train_all --resume './tf_checkpoint.pth.tar'
