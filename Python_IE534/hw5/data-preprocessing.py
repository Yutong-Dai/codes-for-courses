'''
File: data-preprocessing.py
Author: Yutong Dai (rothdyt@gmail.com)
File Created: Monday, 2018-10-08 01:42
Last Modified: Monday, 2018-10-08 01:43
--------------------------------------------
Desscription: data preporcessing for hw5.
'''

import utils
import h5py
import numpy as np
import time

train_dict, db = utils.create_database()
img_all = [db[i] for i in db.keys()]
img_all = [item for l in img_all for item in l]
img_triplet = []
label_triplet = []
print("Begin to sampling. It takes a while...")
start = time.time()
for url in img_all:
    img_tri, label_tri = utils.sample_the_triplet(url, db, train_dict)
    img_triplet.append(img_tri)
    label_triplet.append(label_tri)
end = time.time()
print("Finished in {} mins".format((end-start) / 60))

hf = h5py.File('data_lzf.h5', 'w')
hf.create_dataset('img_triplet', data=img_triplet, compression="lzf")
hf.create_dataset('label_triplet', data=label_triplet, compression="lzf")
hf.close()
