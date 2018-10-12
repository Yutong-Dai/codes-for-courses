'''
File: utils.py
Author: Yutong Dai (rothdyt@gmail.com)
File Created: Sunday, 2018-10-07 19:17
Last Modified: Sunday, 2018-10-07 19:17
--------------------------------------------
Desscription: Helper functions for hw5 (Learning Fine-grained Image Similarity with Deep Ranking).
'''
import os
import numpy as np
import copy
from PIL import Image
import random
import time
import torch.utils.data as data


def create_database():
    """
    Ceate database for tiny-ImageNet.

    @input:
        NA

    @output:
        train_dict: A dictionay contains internal-encoding and its corresponding label.
                    e.g. {'n01443537': 'goldfish, Carassius auratus'}
        db: A dictionay contains internal-encode and all the image address.
            e.g. {'n01443537': ['../data/tiny-imagenet-200/train/n01443537/images/n01443537_0.JPEG']}
    """
    label = [name for name in os.listdir("../data/tiny-imagenet-200/train/") if name != '.DS_Store']
    with open("../data/tiny-imagenet-200/words.txt") as f:
        content = f.readlines()
    content = [x.strip() for x in content]
    full_dict = {}
    for item in content:
        temp = item.split("\t")
        full_dict[temp[0]] = temp[1]
    train_dict = {}
    for idx, name in enumerate(label):
        train_dict[name] = [idx, full_dict[name]]
    db = {}
    for key in train_dict.keys():
        all_imgs = os.listdir("../data/tiny-imagenet-200/train/{}/images".format(key))
        db[key] = ["../data/tiny-imagenet-200/train/{}/images/{}".format(key, i) for i in all_imgs]
    return train_dict, db


def sample_the_triplet(query_img, database, train_dict):
    """
    Given the query to generate a triplet.

    @input:
        query_img: a `key` in the `category_dict`
        database: a dictionay contains all images' paths

    @output:
        A list of the form [query, postive_sample, negative_sample].
        Each of these three is a numpy array.

    @input example:
        query:'../data/tiny-imagenet-200/train/n01443537/images/n01443537_0.JPEG'
        database: {'n01443537': [../data/tiny-imagenet-200/train/n01443537/images/n01443537_0.JPEG]}
    """
    query_img_label = query_img.split("/")[-3]
    postive_pools = copy.deepcopy(database[query_img_label])
    postive_pools.remove(query_img)
    postive_index = random.randint(0, len(postive_pools)-1)
    postive_sample = postive_pools[postive_index]
    negative_pools_label = [i for i in database.keys() if i != query_img_label]
    negative_index_label = random.randint(0, len(negative_pools_label)-1)
    negtive_pools = database[negative_pools_label[negative_index_label]]
    negative_index = random.randint(0, len(negtive_pools)-1)
    negative_sample = negtive_pools[negative_index]

    # It is not necessary to change it to array, since later we will use torchvision.transforms.ToTensor, which will automatically
    # Converts a PIL Image or numpy.ndarray (H x W x C) in the range [0, 255] to a torch.FloatTensor of shape (C x H x W) in the range [0.0, 1.0].
    # Also it will facilitate the construction of the dataloader.
    # However, I can not figure out a good to way pickle such a large object ~850MB. The pickling procedure lasts forever and throws a lot of errors.

    # Warning 1: You have to explicitly close all img, otherwise it will give you tons of shitty errors!!!!
    # Warning 2: Some pictures are in f***ing grey scale, which will cause problems for h5py storgae. You need to check this buy using this.
    #            F***king examples: '../data/tiny-imagenet-200/train/n01644900/images/n01644900_75.JPEG'
    # img = Image.open(query_img)
    # if img.mode == "L":
    #     img = img.convert("RGB")
    # query_img = np.asarray(img.copy(), dtype="uint8")
    # img.close()
    # img = Image.open(postive_sample)
    # if img.mode == "L":
    #     img = img.convert("RGB")
    # postive_sample = np.asarray(img.copy(), dtype="uint8")
    # img.close()
    # img = Image.open(negative_sample)
    # if img.mode == "L":
    #     img = img.convert("RGB")
    # negative_sample = np.asarray(img.copy(), dtype="uint8")
    # img.close()
    img_triplet = [query_img, postive_sample, negative_sample]
    label_triplet = [train_dict[query_img_label][0],
                     train_dict[query_img_label][0],
                     train_dict[negative_pools_label[negative_index_label]][0]]
    return img_triplet, label_triplet


class TinyImageNet(data.Dataset):
    def __init__(self, img_triplet, label_triplet, transform=None):
        self.transform = transform
        self.data = img_triplet
        self.target = label_triplet

    def __getitem__(self, index):
        """
        Args:
            index (int): Index
        Returns:
            tuple: (image, target) where target is index of the target class.
        """
        img_triplet, target_triplet = self.data[index], self.target[index]
        # to return a PIL Image
        img_triplet = [Image.open(img_triplet[i]) for i in range(3)]
        img_triplet = [item.convert("RGB") if item.mode == "L" else item for item in img_triplet]
        if self.transform is not None:
            img_triplet = [self.transform(item) for item in img_triplet]
        return img_triplet, target_triplet

    def __len__(self):
        return len(self.data)

    def __repr__(self):
        return "Triplet for 200-TinyImageNet"


def generate_training_data_set_for_current_epoch():
    train_dict, db = create_database()
    img_all = [db[i] for i in db.keys()]
    img_all = [item for l in img_all for item in l]
    img_triplet = []
    label_triplet = []
    print("Begin to sampling. It takes a while...")
    start = time.time()
    for url in img_all:
        img_tri, label_tri = sample_the_triplet(url, db, train_dict)
        img_triplet.append(img_tri)
        label_triplet.append(label_tri)
    end = time.time()
    print("Finished in {} mins".format((end-start) / 60))
    return img_triplet, label_triplet


def calculateDistance(i1, i2):
    """
    """
    return np.sum((i1-i2)**2)
