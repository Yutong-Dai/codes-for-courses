#!/bin/bash
#PBS -l nodes=1:ppn=16:xk
#PBS -N hw5
#PBS -l walltime=48:00:00
#PBS -e $PBS_JOBNAME.$PBS_JOBID.err
#PBS -o $PBS_JOBNAME.$PBS_JOBID.out
#PBS -m bea
#PBS -M yutongd3@illinois.edu
cd /u/training/tra380/IE534/hw5
. /opt/modules/default/init/bash
module load bwpy/2.0.0-pre1
module load cudatoolkit
aprun -n 1 -N 1 python3.6 main.py --net 'resnet50' --num_epochs 10 --batch_size 15 --train_all --resume './hw5_checkpoint.pth.tar'