'''
File: bb_perf.py
Author: Yutong Dai (rothdyt@gmail.com)
File Created: 2020-03-16 17:40
Last Modified: 2020-03-17 22:41
--------------------------------------------
Description:
'''
import coinor.grumpy as gpy
from coinor.grumpy import BBTree
from coinor.grumpy import MOST_FRACTIONAL, FIXED_BRANCHING, PSEUDOCOST_BRANCHING
from coinor.grumpy import DEPTH_FIRST, BEST_FIRST, BEST_ESTIMATE
import math
import time

branch_strategy = [MOST_FRACTIONAL, FIXED_BRANCHING, PSEUDOCOST_BRANCHING]
search_strategy = [DEPTH_FIRST, BEST_FIRST, BEST_ESTIMATE]
# test problem, (num_vars,num_cons,seed)
problem = [(10, 5, 0),
           (10, 10, 1),
           (10, 15, 2),
           (20, 15, 3),
           (20, 20, 4),
           (20, 25, 5),
           (30, 25, 6),
           (30, 30, 7),
           (30, 35, 8),
           (40, 35, 8),
           (40, 40, 9),
           (40, 45, 0),
           (50, 45, 1),
           (50, 50, 2),
           (50, 55, 3),
           (60, 55, 4),
           (60, 60, 5),
           (60, 65, 6),
           (70, 65, 7),
           (70, 70, 8),
           (70, 75, 9),
           (80, 75, 0),
           (80, 80, 1),
           (80, 85, 2)
           ]

column_titles = '{prob:^15s}   {b:^14}     {s:^16}     {time:^6s}    {size:^7s} \n'.format(prob="Problem", b="Branch", s="Search",
                                                                                           time="Time", size="Size")
with open('profile.txt', "a") as logfile:
    logfile.write(column_titles)
for p in problem:
    var, con, seed = p
    problem_name = "{}-{}-{}".format(var, con, seed)
    CONSTRAINTS, VARIABLES, OBJ, MAT, RHS = gpy.GenerateRandomMIP(numVars=var,
                                                                  numCons=con,
                                                                  rand_seed=seed)
    for b in branch_strategy:
        for s in search_strategy:
            T = BBTree()
            time_start = time.time()
            solution, bb_optimal = gpy.BranchAndBound(T, CONSTRAINTS, VARIABLES, OBJ,
                                                      MAT, RHS,
                                                      branch_strategy=b,
                                                      search_strategy=s)
            time_elpased = time.time() - time_start
            tree_size = T._lp_count
            content = "{prob:^15s}   {b:<20s}  {s:<16s} {time:^5.3e}  {size:^6d} \n".format(prob=problem_name, b=b.replace(" ", "_"), s=s.replace(" ", "_"), time=time_elpased, size=tree_size)
            with open('profile.txt', "a") as logfile:
                logfile.write(content)
