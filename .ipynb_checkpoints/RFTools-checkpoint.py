#!/usr/bin/env python
# -*- coding: utf-8 -*-
# @Date    : 2018-07-07 17:56:21
# @Author  : Roth (rothdyt@gmail.com)
# @Version : 0.9
# @Last Modified: 2018-08-18

import pydotplus
import re 
import pickle
from sklearn import tree
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import pandas as pd
import logging

"""
Here, we provide two ways to find the path from root node to terminal node that gives the maximum prediction.
First method `return_node_path_to_max_prediction` is based on the pydotplus.graph_from_dot_data to translate dot file into a graph object, 
which has many attributes for plotting. But these consumes a tremendous a mount of time.

The scecond method is to define a class called `Graph()` to process dot file produced by the `tree.export_graphviz` from scikit-learn. This 
method significantly reduces the parsing time.

"""

def return_node_path_to_max_prediction(onetree, verbose=True):
    """
    @input: a tree from the sklearn randomforest
    @output: the node path to maxmium terminal node
        [[split_node_1], [split_node_2], ...]
        [splite_node_1] = [var_index, cutoff, direction]
    """
    if verbose:
        print("Generating Tree Graph, it may take a while...")
    dot_data = tree.export_graphviz(onetree,
                                    out_file = None,
                                    filled   = True,
                                    rounded  = True,
                                    special_characters = True)  
    graph = pydotplus.graph_from_dot_data(dot_data)
    graph_ = {}
    for edge in graph.get_edge_list():
        graph_[edge.get_source()] = edge.get_destination()
    # find all terminal node
    terminal_node = {}
    non_decimal = re.compile(r'[^\d.]+')
    for node in graph.get_node_list():
        if node.get_name() not in graph_:
            if node.get_name() not in ["node", "edge"]:
                value = node.get_label()
                value = re.sub(r'.*v', 'v', value)
                terminal_node[node.get_name()] = float(non_decimal.sub('', value))
    # find the path down to the terminal with maximum predition value
    flag = True
    destination = max(terminal_node, key=terminal_node.get)
    edge_list = graph.get_edge_list()
    node_list = graph.get_node_list()
    split_node = []
    while flag:
        myedge = [edge for edge in edge_list  if edge.get_destination() == destination][0]
        if int(myedge.get_destination()) - int(myedge.get_source()) > 1:
            direction = "Right"
        else:
            direction = "Left"
        
        mynode = [node for node in node_list if node.get_name() == myedge.get_source()][0]
        var_val = re.findall(r"[-+]?\d*\.\d+|\d+", mynode.get_label())[:2]
        # record the growing path:
        #  var_val[0]: Index of variable participating in splitting
        #  var_val[1]: cutoff point of the splitting
        #  direction: If Right, means greater than var_val[1]; 
        #             If Left, means no greater than var_val[1]
        split_node.append([int(var_val[0]),float(var_val[1]),direction])
        if verbose:
            print(myedge.get_destination() + "<-" + myedge.get_source() + 
                  ": Split at Variable X" + var_val[0] + "; The cutoff is " + var_val[1] + 
                 "; Turn " + direction)
        destination = myedge.get_source()
        if destination == "0":
            flag = False
    return [*reversed(split_node)]


class Graph:

    def __init__(self, dot_data, genre = "data"):
        """
        store the dot data.
        """
        if genre == "data":
            graph = dot_data.split("\n")
        elif genre == "file":
            graph = []
            with open(dot_file, 'r') as f:
                graph = f.readlines()
        else:
            raise TypeError("Invalid File Type!")

        graph_ = []
        for i in graph:
            graph_.append(i.split(';'))        
        self.raw_graph = graph_
        
    
    def get_graph(self):
        """
        Construct trees fromdot file in human-readable structure.
        
        Output: a dictionary of the format:
            {parenr node : [left child node, right child node, splitting variable index, cutoff],
             parenr node : [left child node, right child node, splitting variable index, cutoff],
             ...
            }
        """
        graph__ = {}
        for i in self.raw_graph:
            info = []
            for j in i:
                node_list = re.findall(r'[0-9][0-9]* -> [0-9][0-9]*', j)
                if node_list != []:
                    edge = node_list[0]
                    source = re.findall('[0-9][0-9]* ', edge)[0][:-1]
                    target = re.findall(' [0-9][0-9]*', edge)[0][1:]
                    if source not in graph__:
                        graph__[source] = [target]
                    else:
                        graph__[source].append(target)
        for key, val in self.get_none_terminal_nodes().items():
            graph__[key].append(val[0])
            graph__[key].append(val[1])
        return graph__ 
        
    def get_none_terminal_nodes(self):
        """
        Find all none terminal nodes.
        Return a dictionary of the form:
        {"node name": [variable name(index), cutoff value]}
        """
        non_terminal_nodes = {}
        for i in self.raw_graph:
            info = []
            for j in i:
                if 'X<SUB>' in j:
                    info.append(j)
                if 'value' in j and 'mse' in j:
                    info.append(j)
                if len(info) == 2:
                    node_var_cutoff = re.findall(r"[-+]?\d*\.\d+|\d+", info[0])
                    node_var_cutoff.append(re.findall(r"[-+]?\d*\.\d+|\d+", info[1])[0])
                    non_terminal_nodes[node_var_cutoff[0]] = [node_var_cutoff[1], node_var_cutoff[2]]
                    break
        return non_terminal_nodes
    
    def get_terminal_nodes(self):
        """
        Find all terminal nodes.
        
        Output: A list of terminal nodes with each element looks like:
            'node_name [label=<mse = foo<br/>samples = foor <br/>value = foo>] ', for example
            '7 [label=<mse = 0.0<br/>samples = 2<br/>value = 2.8>] '
        """
        terminal_nodes = []
        for i in self.raw_graph:
            for j in i:
                if 'label' in j and '->' not in j and 'mse' in j:
                    terminal_nodes.append(j)
        return terminal_nodes
    
    def get_max_prediction_terminal_node(self):
        """
        Find the terminal node that gives the max prediction.
        
        Output:
            {node name : predicted value}
        """
        terminal_nodes = self.get_terminal_nodes()
        non_decimal = re.compile(r'[^\d.]+')
        terminal_nodes_ = {}
        for node in self.get_terminal_nodes():
            terminal_nodes_[re.search(r'[0-9][0-9]*', node).group()] = float( non_decimal.sub("", re.sub(r'.*v', 'v', node)))
        max_node_name = max(terminal_nodes_, key=terminal_nodes_.get)
        return {max_node_name: terminal_nodes_[max_node_name]}
    
    def get_path_to_max_prediction_terminal_node(self, verbose=False):
        """
        Find the path from root node to the selected terminal node with max prediction. If vebose = True, then the bottom up path is printed out.
        
        Output:
            A list of lists in the form of
                [variable name, cutoff value, direction]
            ,which is sorted from root node to terminal node.
        """
        destination = [*self.get_max_prediction_terminal_node().keys()][0]
        graph = self.get_graph()
        path = []
        #nodes = []
        for i in  range(len(graph.keys())):
            #nodes.append(destination)
            for key in graph.keys():
                if destination in graph[key][:2]:
                    key_to_del = key
                    if destination == graph[key][0]:
                        path.append([int(graph[key][2]), float(graph[key][3]), "Left"])
                    else:
                        path.append([int(graph[key][2]), float(graph[key][3]), "Right"])
                    break
            if verbose :
                print("Parent Node:{}, Child Node:{}, Split at variable X{}, the cutoff is {}, Turn {}.".format(key_to_del, destination, path[i][0], path[i][1], path[i][2] ))
            
            destination = key_to_del
            try:
                del graph[key_to_del]
            except KeyError:
                pass
            if destination == "0":
                break
        return [*reversed(path)]

############################################
# Functions to process the returned paths. #
############################################

def collect_path(rf, method="quick", verbose=True):
    """
    Collect paths from RandomForest objects. This function is the most time-consuming part.
    
    Output:
        A list of outputs from get_path_to_max_prediction_terminal_node.
    """
    n_tree = len(rf)
    result = []
    if method == "quick":
        for i in range(n_tree):
            if verbose:
                if (i+1) % 100 == 0:
                    print("Construct the %s tree graph out of %s trees" %(i+1, n_tree))
                dot_data = tree.export_graphviz(rf.estimators_[i], out_file = None, rounded  = True, special_characters = True)  
                G = Graph(dot_data)
                result.append(G.get_path_to_max_prediction_terminal_node())
    else:
        result.append(return_node_path_to_max_prediction(rf.estimators_[i], verbose=False))
    return result

    
def collect_managment(collected_path, managments):
    """
    Derive the intervals for each managmennt.
    
    Output:
        A dictionary for each managment of the form:
         {"variable":[[collection of left endpoints], [collection of right endpoints]]}
        For example:
        {'N.total': [[33.0,..,],[55.125,....]]
    """
    managment_range = {k: [[] for _ in range(2)] for k in managments}
    for i in range(len(collected_path)):
        for j in range(len(collected_path[i])):
            if collected_path[i][j][2] == "Left":
                 managment_range[managments[collected_path[i][j][0]]][1].append(collected_path[i][j][1])
            else:
                managment_range[managments[collected_path[i][j][0]]][0].append(collected_path[i][j][1])

    return (managment_range)

#####################################################################
# quantile verison of summary_continuous variables; discard for now.#
#####################################################################

# def summary_continuous(var_list, managment_range, offset_l=5, offset_r=0, verbose=False, figure="Idaho.jpg", num_cont=9, fig_cols=3):
#     results = {}
#     fig_index = 0
#     fig_rows = round(num_cont / fig_cols) + 1
#     plt.figure(figsize=(20,20))
#     for cont_var_name in var_list[:num_cont]:
#         lower = managment_range[cont_var_name][0]
#         upper = managment_range[cont_var_name][1]
#         if len(lower) > 0 and len(upper) >0:
#             find_min_quantile = False
#             upper.sort(); lower.sort()
#             lower_ = [*set(lower)]; lower_.sort()
#             for j in range(len(lower_)):
#                 for i in range(len(upper)):
#                     if upper[i] > lower_[-(j+1)]:
#                         min_quantile = ((i+1) / len(upper)) * 100
#                         find_min_quantile = True
#                         break
#                 minimal_left_offset = (j / len(lower)) * 100
#                 if find_min_quantile:
#                     break
#             if (abs(minimal_left_offset - 0) > 1e-6):
#                 print("Additional {} offset enforced for {}.".format(round(minimal_left_offset), cont_var_name))      
#             left_end = np.percentile(lower,100 -minimal_left_offset - offset_l)
#             right_end = np.percentile(upper,min_quantile + offset_r)
#             interval = [left_end, right_end]
#             if verbose:
#                 print("The recommended interval for {} is [{},{}]".format(cont_var_name, left_end, right_end))
#             results[cont_var_name] = interval
#             if figure is not None:
#                 fig_index += 1
#                 plt.subplot(fig_rows, fig_cols, fig_index)
#                 plt.plot(lower, np.zeros_like(lower)-0.02,"r*", label="Collection of Left Endpoints")
#                 plt.plot(upper, np.zeros_like(upper)+0.02,"b*", label="Collection Right Endpoints")
#                 plt.plot((left_end,right_end), (0,0),"k-", linewidth=2)
#                 plt.plot((left_end,left_end), (-0.005, 0.005), "r-", linewidth=2, 
#                          label=str(left_end) + " with {}% left offset".format(round(offset_l + minimal_left_offset)))
#                 plt.plot((right_end, right_end), (-0.005, 0.005), "b-", linewidth=2,
#                          label=str(right_end) + " with {}% right offset".format(round(min_quantile + offset_r)))
#                 plt.ylim(-0.03,0.04)#;plt.xlim(left_end*0.5, right_end*1.5)
#                 plt.xlabel(cont_var_name, fontsize=15); plt.yticks([])
#                 plt.legend()
#         elif len(lower) == 0 and len(upper) > 0:
#             print("Left offset for {} is disabled.".format(cont_var_name))
#             left_end = 0 # for plot, not real
#             right_end = np.percentile(upper, offset_r)
#             interval = ["Unknown", right_end]
#             if verbose:
#                 print("The recommended interval for {} is [{},{}]".format(cont_var_name, "Unknown", right_end))
#             results[cont_var_name] = interval
#             if figure is not None:
#                 fig_index += 1
#                 plt.subplot(fig_rows, fig_cols, fig_index)
#                 plt.plot(upper, np.zeros_like(upper)+0.02,"b*", label="Collection Right Endpoints")
#                 plt.plot((left_end,right_end), (0,0),"k-.", linewidth=2)
#                 plt.plot((right_end, right_end), (-0.005, 0.005), "b-", linewidth=2,
#                          label=str(right_end) + " with {}% right offset".format(round(offset_r)))
#                 plt.text(-0.1, 0.01, "Left Endpoint Unknown", bbox=dict(facecolor='red', alpha=0.5),fontsize=12, rotation='vertical')
#                 plt.ylim(-0.03,0.04)#; plt.xlim(-right_end, right_end*1.5)
#                 plt.xlabel(cont_var_name, fontsize=15); plt.yticks([])
#                 plt.legend()

#         elif len(lower) > 0 and len(upper) == 0:
#             print("Right offset is disabled.")
#             left_end = np.percentile(lower, 100 - offset_l)
#             right_end = left_end * 1.5 # for plot, not real
#             interval = ["Unknown", right_end]
#             if verbose:
#                 print("The recommended interval for {} is [{},{}]".format(cont_var_name, left_end, "Unknown"))
#             results[cont_var_name] = interval
#             if figure is not None:
#                 fig_index += 1
#                 plt.subplot(fig_rows, fig_cols, fig_index)
#                 plt.plot(lower, np.zeros_like(lower)-0.02,"r*", label="Collection Left endpoints")
#                 plt.plot((left_end,right_end), (0,0),"k-.", linewidth=2)
#                 plt.plot((left_end, left_end), (-0.005, 0.005), "r-", linewidth=2,
#                          label=str(left_end) + " with {}% left offset".format(round(offset_l)))
#                 plt.text(right_end, 0.01, "Right Endpoint Unknown", bbox=dict(facecolor='red', alpha=0.5),fontsize=12, rotation='vertical')
#                 plt.ylim(-0.03,0.04);
#                 plt.xlabel(cont_var_name, fontsize=15); plt.yticks([])
#                 plt.legend()
#         else:
#             interval = ["Unknown", "Unknown"]
#             if verbose:
#                 print("The recommended interval for {} is [{},{}]".format(cont_var_name, "Unknown", "Unknown"))
#             results[cont_var_name] = interval
#             if figure is not None:
#                 fig_index += 1
#                 plt.subplot(fig_rows, fig_cols, fig_index)            
#                 plt.text(0, 0, "No data available to plot", fontsize=20, horizontalalignment='center', verticalalignment='center')
#                 plt.ylim(-1,1); plt.xlim(-1,1)
#                 plt.xlabel(cont_var_name, fontsize=15)
#                 plt.yticks([]); plt.xticks([])
#     plt.savefig(figure)
#     plt.close()
#     return results

# find range by doing the resampling
def summary_continuous(var_list, managment_range, sample_size=500, num_cont=9, range_percentile=10, verbose=False):
    intervals_final_all = {}
    for var_name in var_list[:num_cont]:
        lower = managment_range[var_name][0]
        upper = managment_range[var_name][1]
        if len(lower) > 0 and len(upper) > 0:
            endpoints = []
            intervals_samples = []
            intervals_final = []
            count = 0
            for i in range(500):
                np.random.seed(i+1)
                l = np.random.choice(managment_range[var_name][0]);
                np.random.seed(i+2)
                u = np.random.choice(managment_range[var_name][1]);
                if l < u:
                    intervals_samples.append({"lower":l, "upper":u})
                    endpoints.append(l)
                    endpoints.append(u)
                else:
                    count += 1       
            ends = [*set(endpoints)]
            if (len(ends) <= 1):
                upper_bound = "Invalid"
                lower_bound = "Invalid"
                intervals_final_all[var_name] = [lower_bound, upper_bound]
            else:
                ends.sort()
                for i in range(len(ends)-1):
                    intervals_final.append({"lower":ends[i], "upper":ends[i+1], "count":0})
                for interval in intervals_samples:
                    for item in intervals_final:
                        if item["lower"] >= interval["lower"] and  item["upper"] <= interval["upper"]:
                            item["count"] += 1
                count_distribution = [item["count"] for item in intervals_final]
                if count > max(count_distribution):
                    upper_bound = "Invalid"
                    lower_bound = "Invalid"
                    intervals_final_all[var_name] = [lower_bound, upper_bound]
                else:
                    ratio = np.array(count_distribution) / sum(count_distribution)
                    ratio_reversed = ratio[::-1]
                    for i in range(len(ratio)):
                        if np.cumsum(ratio[:(i+1)])[-1] >= range_percentile/100:
                            lower_bound = intervals_final[i]["upper"]
                            break
                    for i in range(len(ratio)):
                        if np.cumsum(ratio_reversed[:(i+1)])[-1] >= range_percentile/100:
                            upper_bound = intervals_final[-(i+1)]["lower"]
                            break
                    intervals_final_all[var_name] = [lower_bound, upper_bound]
        elif len(lower) == 0 and len(upper) > 0:
            upper_bound = np.percentile(upper, range_percentile)
            lower_bound = 0
            intervals_final_all[var_name] = [lower_bound, upper_bound]
        elif len(lower) > 0 and len(upper) == 0:
            upper_bound = "Unknown"
            lower_bound = np.percentile(lower, range_percentile)
            intervals_final_all[var_name] = [lower_bound, upper_bound]
        else:
            upper_bound = "Unknown"
            lower_bound = "Unknown"
            intervals_final_all[var_name] = [lower_bound, upper_bound]
        if verbose:
            print("The recommended interval for {} is [{},{}]".format(var_name, lower_bound, upper_bound))
    return intervals_final_all

def visualize_cont_intervals(results_cont, managment_range, var_list, figure_name="Idaho.jpg", num_cont=9, fig_cols = 3):
    total = len(var_list[num_cont:])
    fig_rows = round(total / fig_cols) + 1
    fig = plt.figure(figsize=(20,40))
    fig_index = 0
    for var_name in var_list[:num_cont]:
        lower = managment_range[var_name][0]
        upper = managment_range[var_name][1]
        left_end = results_cont[var_name][0]
        right_end = results_cont[var_name][1]
        fig_index += 1
        ax = fig.add_subplot(fig_rows, fig_cols, fig_index)      
        if len(lower)>0 and len(upper)>0:
            if left_end == "Invalid":
                ax.text(0, 0, "Unavailable based on the data", fontsize=20, horizontalalignment='center', verticalalignment='center')
                ax.set_xlim((-1,1)); ax.set_ylim((-1,1)); ax.set_yticks([]); ax.set_xticks([])
            else:
                ax.plot(lower, np.zeros_like(lower)-0.02,"r*", label="Collection of Left Endpoints")
                ax.plot(upper, np.zeros_like(upper)+0.02,"b*", label="Collection Right Endpoints")
                ax.plot((left_end,right_end), (0,0),"k-", linewidth=2)
                ax.plot((left_end,left_end), (-0.005, 0.005), "r-", linewidth=2, label=str(left_end))
                ax.plot((right_end, right_end), (-0.005, 0.005), "b-", linewidth=2,label=str(right_end))
                ax.plot((left_end,right_end), (0,0),"k-", linewidth=2)
                ax.set_ylim(-0.03,0.04); ax.set_yticks([])
                ax.legend(loc="upper right")
                ax.set_title(var_name)
        elif len(lower)==0 and len(upper)>0:
            ax.plot(upper, np.zeros_like(upper)+0.02,"b*", label="Collection Right Endpoints")
            ax.plot((left_end,left_end), (-0.005, 0.005), "r-", linewidth=2, label=str(left_end))
            ax.plot((right_end, right_end), (-0.005, 0.005), "b-", linewidth=2,label=str(right_end))
            ax.plot((left_end,right_end), (0,0),"k-", linewidth=2)
            ax.set_ylim(-0.03,0.04); ax.set_yticks([])
            ax.legend(loc="upper right")
            ax.set_title(var_name)
        elif len(lower)>0 and len(upper)==0:
            ax.plot(lower, np.zeros_like(lower)-0.02,"b*", label="Collection Right Endpoints")
            ax.plot((left_end,left_end), (-0.005, 0.005), "r-", linewidth=2, label=str(left_end))
            ax.plot((right_end, right_end), (-0.005, 0.005), "b-", linewidth=2,label=str("Unknown"))
            ax.set_ylim(-0.03,0.04); ax.set_yticks([])
            ax.legend(loc="upper right")
            ax.set_title(var_name)
        else:
            ax.text(0, 0, "No data available to plot", fontsize=20, horizontalalignment='center', verticalalignment='center')
            ax.set_xlim((-1,1)); ax.set_ylim((-1,1)); ax.set_yticks([]); ax.set_xticks([])
            ax.set_title(var_name)
    fig.savefig(figure_name)
    plt.close()
    
def summary_categorical(var_list, managments, managment_range, num_cont=9):
    cat_level_count_summary = {}
    for cat in var_list[num_cont:]:
        levels = []
        for item in managments[num_cont:]:
            if item.split("_")[0] == cat:
                levels.append(item)
        if  set(["Yes", "No"]).intersection([levels[i].split("_")[1] for i in range(len(levels))]):
            cat_level_count_summary[cat] = {"Yes":0, "No":0}
            for level in levels:
                if level.split("_")[1] == "No":
                    cat_level_count_summary[cat]["Yes"] += len(managment_range[level][1])
                    cat_level_count_summary[cat]["No"] += len(managment_range[level][0])
                else:
                    cat_level_count_summary[cat]["Yes"] += len(managment_range[level][0])
                    cat_level_count_summary[cat]["No"] += len(managment_range[level][1])
                
        else:
            cat_level_count_summary[cat] = {}
            for level in levels:
                level_counts = managment_range[level]
                num_of_splits = len(level_counts[0]) + len(level_counts[1])
                try:
                    yes_prob = len(level_counts[0]) / num_of_splits
                except ZeroDivisionError:
                    yes_prob = 0
                cat_level_count_summary[cat][level] = {"Yes Probability":yes_prob, "Number of splits":num_of_splits}
        
    return cat_level_count_summary

def light(x):
    if x<0.3:
        return "red"
    elif x<0.5:
        return "yellow-no"
    elif x<0.7:
        return "yellow-do"
    else:
        return "green"

def visualize_cat_level_count_summary(cat_level_count_summary, var_list, figure_name="Idaho.jpg", num_cont=9, fig_cols = 3):        
    total = len(var_list[num_cont:])
    fig_rows = round(total / fig_cols) + 1
    fig = plt.figure(figsize=(20,20))
    fig_index = 0
    reports = {}
    for var in var_list[num_cont:]:
        fig_index += 1
        if var in ["S.indicator", "Zn.indicator", "SeedTreated"]:
            item = cat_level_count_summary[var]
            try:
                counts = [item["Yes"], item["No"]]
                probs = [counts[0]/sum(counts), counts[1]/sum(counts)]
            except ZeroDivisionError:
                probs = [0, 0]
            labels = ["Yes", "No"]
            cmap = plt.get_cmap('rainbow')
            colors = [cmap(i) for i in np.linspace(0, 1, len(set(labels)))]
            df = pd.DataFrame(dict(count=counts, prob=probs, label=labels, color=colors))
        else:
            labels = []
            counts = []
            probs = []
            for key, value in cat_level_count_summary[var].items():
                labels.append(key.split("_")[1])
                counts.append(value['Number of splits'])
                probs.append(value['Yes Probability'])
            cmap = plt.get_cmap('rainbow')
            colors = [cmap(i) for i in np.linspace(0, 1, len(set(labels)))]
            df = pd.DataFrame(dict(count=counts, prob=probs, label=labels, color=colors))
        threshold = max(sum(df["count"]) * 0.1, 30)
        temp = df[["count","label","prob"]].loc[df["count"]>=threshold]
        try:
            temp['light'] = temp.apply(lambda x: light(x.prob), axis=1)
        except ValueError:
            threshold = sum(df["count"]) * 0.1
            temp = df[["count","label","prob"]].loc[df["count"]>=threshold]
            temp['light'] = temp.apply(lambda x: light(x.prob), axis=1)
        reports[var] = temp.drop(["count", "prob"], axis=1)
        ax = fig.add_subplot(fig_rows, fig_cols, fig_index)
        
        for x,y,c,l in zip(df["count"], df["prob"], df["color"],df["label"]):
            ax.scatter(x,y,color=c,label=l)
        ax.legend(loc="upper left")
        ax.set_title(var)
        ax.hlines(y=0.3, xmin=0, xmax=max(df["count"]), colors= "red", linestyles="dotted")
        ax.hlines(y=0.7, xmin=0, xmax=max(df["count"]), colors= "green", linestyles="dotted")
        ax.vlines(x=threshold, ymin=0, ymax=1, colors= "red", linestyles="dotted")
    fig.savefig(figure_name)
    plt.close()
    return reports

#####################################################################
# quantile verison of summary_continuous variables; discard for now.#
#####################################################################

# def summary_continuous(var_list, managment_range, offset_l=5, offset_r=0, verbose=False, figure="Idaho.jpg", num_cont=9, fig_cols=3):
#     results = {}
#     fig_index = 0
#     fig_rows = round(num_cont / fig_cols) + 1
#     plt.figure(figsize=(20,20))
#     for cont_var_name in var_list[:num_cont]:
#         lower = managment_range[cont_var_name][0]
#         upper = managment_range[cont_var_name][1]
#         if len(lower) > 0 and len(upper) >0:
#             find_min_quantile = False
#             upper.sort(); lower.sort()
#             lower_ = [*set(lower)]; lower_.sort()
#             for j in range(len(lower_)):
#                 for i in range(len(upper)):
#                     if upper[i] > lower_[-(j+1)]:
#                         min_quantile = ((i+1) / len(upper)) * 100
#                         find_min_quantile = True
#                         break
#                 minimal_left_offset = (j / len(lower)) * 100
#                 if find_min_quantile:
#                     break
#             if (abs(minimal_left_offset - 0) > 1e-6):
#                 print("Additional {} offset enforced for {}.".format(round(minimal_left_offset), cont_var_name))      
#             left_end = np.percentile(lower,100 -minimal_left_offset - offset_l)
#             right_end = np.percentile(upper,min_quantile + offset_r)
#             interval = [left_end, right_end]
#             if verbose:
#                 print("The recommended interval for {} is [{},{}]".format(cont_var_name, left_end, right_end))
#             results[cont_var_name] = interval
#             if figure is not None:
#                 fig_index += 1
#                 plt.subplot(fig_rows, fig_cols, fig_index)
#                 plt.plot(lower, np.zeros_like(lower)-0.02,"r*", label="Collection of Left Endpoints")
#                 plt.plot(upper, np.zeros_like(upper)+0.02,"b*", label="Collection Right Endpoints")
#                 plt.plot((left_end,right_end), (0,0),"k-", linewidth=2)
#                 plt.plot((left_end,left_end), (-0.005, 0.005), "r-", linewidth=2, 
#                          label=str(left_end) + " with {}% left offset".format(round(offset_l + minimal_left_offset)))
#                 plt.plot((right_end, right_end), (-0.005, 0.005), "b-", linewidth=2,
#                          label=str(right_end) + " with {}% right offset".format(round(min_quantile + offset_r)))
#                 plt.ylim(-0.03,0.04)#;plt.xlim(left_end*0.5, right_end*1.5)
#                 plt.xlabel(cont_var_name, fontsize=15); plt.yticks([])
#                 plt.legend()
#         elif len(lower) == 0 and len(upper) > 0:
#             print("Left offset for {} is disabled.".format(cont_var_name))
#             left_end = 0 # for plot, not real
#             right_end = np.percentile(upper, offset_r)
#             interval = ["Unknown", right_end]
#             if verbose:
#                 print("The recommended interval for {} is [{},{}]".format(cont_var_name, "Unknown", right_end))
#             results[cont_var_name] = interval
#             if figure is not None:
#                 fig_index += 1
#                 plt.subplot(fig_rows, fig_cols, fig_index)
#                 plt.plot(upper, np.zeros_like(upper)+0.02,"b*", label="Collection Right Endpoints")
#                 plt.plot((left_end,right_end), (0,0),"k-.", linewidth=2)
#                 plt.plot((right_end, right_end), (-0.005, 0.005), "b-", linewidth=2,
#                          label=str(right_end) + " with {}% right offset".format(round(offset_r)))
#                 plt.text(-0.1, 0.01, "Left Endpoint Unknown", bbox=dict(facecolor='red', alpha=0.5),fontsize=12, rotation='vertical')
#                 plt.ylim(-0.03,0.04)#; plt.xlim(-right_end, right_end*1.5)
#                 plt.xlabel(cont_var_name, fontsize=15); plt.yticks([])
#                 plt.legend()

#         elif len(lower) > 0 and len(upper) == 0:
#             print("Right offset is disabled.")
#             left_end = np.percentile(lower, 100 - offset_l)
#             right_end = left_end * 1.5 # for plot, not real
#             interval = ["Unknown", right_end]
#             if verbose:
#                 print("The recommended interval for {} is [{},{}]".format(cont_var_name, left_end, "Unknown"))
#             results[cont_var_name] = interval
#             if figure is not None:
#                 fig_index += 1
#                 plt.subplot(fig_rows, fig_cols, fig_index)
#                 plt.plot(lower, np.zeros_like(lower)-0.02,"r*", label="Collection Left endpoints")
#                 plt.plot((left_end,right_end), (0,0),"k-.", linewidth=2)
#                 plt.plot((left_end, left_end), (-0.005, 0.005), "r-", linewidth=2,
#                          label=str(left_end) + " with {}% left offset".format(round(offset_l)))
#                 plt.text(right_end, 0.01, "Right Endpoint Unknown", bbox=dict(facecolor='red', alpha=0.5),fontsize=12, rotation='vertical')
#                 plt.ylim(-0.03,0.04);
#                 plt.xlabel(cont_var_name, fontsize=15); plt.yticks([])
#                 plt.legend()
#         else:
#             interval = ["Unknown", "Unknown"]
#             if verbose:
#                 print("The recommended interval for {} is [{},{}]".format(cont_var_name, "Unknown", "Unknown"))
#             results[cont_var_name] = interval
#             if figure is not None:
#                 fig_index += 1
#                 plt.subplot(fig_rows, fig_cols, fig_index)            
#                 plt.text(0, 0, "No data available to plot", fontsize=20, horizontalalignment='center', verticalalignment='center')
#                 plt.ylim(-1,1); plt.xlim(-1,1)
#                 plt.xlabel(cont_var_name, fontsize=15)
#                 plt.yticks([]); plt.xticks([])
#     plt.savefig(figure)
#     plt.close()
#     return results

#####################################################################
# quantile method to report categorical vars. Disabled for now. ####
#####################################################################

# def get_levels_to_report(cat_level_count_summary, var_list, num_count=9, cut_percentile=0.1):
#     reports = {}
#     for var_name in var_list[num_count:]:
#         if var_name in ["S.indicator", "Zn.indicator", "SeedTreated"]:
#             S = results_cat[var_name]
#             report={var_name:max(S, key=S.get)}
#         else:
#             mydict = {}
#             for key, val in results_cat[var_name].items():
#                 mydict[key]=val['Number of splits']
#             S = pd.Series(mydict); S /= sum(S)
#             S.sort_values(axis=0, ascending=True, kind='quicksort', na_position='last', inplace=True)
#             S = np.cumsum(S)
#             index = S.loc[S>cut_percentile].index
#             report = {}
#             for i in index:
#                 if results_cat[var_name][i]['Yes Probability'] > 0.5:
#                     report[i] = "Yes"
#                 else:
#                     report[i] = "No"
#             pd.Series(report)
#         reports[var_name] = report
#     return reports
# get_levels_to_report(results_cat, var_list, num_count=9, cut_percentile=0.2)