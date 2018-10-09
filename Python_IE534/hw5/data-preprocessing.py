'''
File: data-preprocessing.py
Author: Yutong Dai (rothdyt@gmail.com)
File Created: Monday, 2018-10-08 01:42
Last Modified: Monday, 2018-10-08 01:43
--------------------------------------------
Desscription: data preporcessing for hw5.
'''

import os
# os.chdir(os.path.dirname(os.path.realpath(__file__)))
category = [name for name in os.listdir("../data/tiny-imagenet-200/train/") if name != '.DS_Store']
with open("../data/tiny-imagenet-200/words.txt") as f:
    content = f.readlines()
content = [x.strip() for x in content]
full_dict = {}
for item in content:
    temp = item.split("\t")
    full_dict[temp[0]] = temp[1]
train_dict = {}
for name in category:
    train_dict[name] = full_dict[name]
print(train_dict)
