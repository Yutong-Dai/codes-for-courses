# This script creates a PBS file that runs one hyperameter setting
# on a single node.

import sys

training_file = sys.argv[1]
walltime = sys.argv[2]
jobname = sys.argv[3]
netid = sys.argv[4]
directory = sys.argv[5]

# Change these if your hyperparameters change
trial_number = sys.argv[6]

print "#!/bin/bash"
print "#PBS -l nodes=1:ppn=16:xk"
print "#PBS -N {0}_{1}".format(jobname,trial_number) # Change this if your hyperparameters change
print "#PBS -l walltime={0}".format(walltime)
print "#PBS -e $PBS_JOBNAME.$PBS_JOBID.err"
print "#PBS -o $PBS_JOBNAME.$PBS_JOBID.out"
print "#PBS -M {0}@illinois.edu".format(netid)

print "cd {0}".format(directory)

print ". /opt/modules/default/init/bash"
print "module load bwpy"
print "module load cudatoolkit"

print "aprun -n 1 -N 1 python {0} {1}".format(training_file, trial_number) # Change this if your hyperameters change
