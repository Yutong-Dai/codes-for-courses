'''
File: utils.py
Author: Yutong Dai (rothdyt@gmail.com)
File Created: Sunday, 2018-10-07 19:17
Last Modified: Sunday, 2018-10-07 19:17
--------------------------------------------
Desscription: Helper functions for hw5 (Learning Fine-grained Image Similarity with Deep Ranking).
'''

import numpy as np


def calculateDistance(i1, i2):
    """a
    """
    return np.sum((i1-i2)**2)
