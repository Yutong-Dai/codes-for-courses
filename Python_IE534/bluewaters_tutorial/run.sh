#!/bin/bash
# This script runs multiple batch jobs to test different hyperparameter
# settings. For each setting, it creates a different PBS file and calls
# it.

# MODIFY THESE
declare training_file="test_python_script.py"
declare walltime="01:00:00"
declare jobname="bluewaters_test"
declare netid="yournetid"
declare directory="~/bluewaters_tutorial/"

# Declare the hyperparameters you want to iterate over
declare -a trial_number=(0 1 2)

# For each parameter setting we generate a new PBS file and run it
for trial in "${trial_number[@]}"
do
  python generate_pbs.py $training_file $walltime $jobname $netid $directory $trial > run.pbs
  echo "Submitting $trial"
  qsub run.pbs -A bauh
done
