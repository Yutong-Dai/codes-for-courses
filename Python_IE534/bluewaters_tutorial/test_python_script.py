import numpy as np
import os
import sys

trials = [
['adam', '0.01', '30'],
['sgd', '0.1', '60'],
['rmsprop', '0.01', '30'],
]

trial_number = int(sys.argv[1])
LR = float(trials[trial_number][1])
opt = trials[trial_number][0]
number_of_epochs = int(trials[trial_number][2])

print(trial_number)
print(LR)
print(opt)
print(number_of_epochs)
