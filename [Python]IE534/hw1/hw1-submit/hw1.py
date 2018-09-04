#!/usr/bin/env python
# -*- coding: utf-8 -*-
# @Date    : 2018-09-02 22:33:26
# @Author  : Yutong Dai
# @Version : 0.9

import numpy as np
import h5py
import time 
import copy



file_name = "./data/MNISTdata.hdf5"
data = h5py.File(file_name, "r")
x_train = np.float32(data["x_train"][:])
y_train = np.int32(np.hstack(np.array(data["y_train"])))
x_test = np.float32(data["x_test"][:])
y_test = np.int32(np.hstack(np.array(data["y_test"])))
data.close()

class MnistModel():
    def __init__(self, x_train, y_train, x_test, y_test, hidden_units=100, learning_rate=0.01, batch_size=20, num_epochs=5, seed=None):
        self.x_train = x_train
        self.x_test = x_test
        self.y_train = y_train
        self.y_test = y_test
        self.num_inputs = self.x_train.shape[1]
        self.num_outputs = 10
        self.hidden_units = hidden_units
        self.learning_rate = learning_rate
        self.batch_size = batch_size
        self.num_epochs = num_epochs
        self.params = {}
        self.gradients = {}
        if seed is not None:
            r = np.random.RandomState(seed)
            self.params["W"] = r.randn(self.hidden_units, self.num_inputs) / np.sqrt(self.num_inputs)
            self.params["b1"] = np.zeros((self.hidden_units, 1))  
            self.params["C"] = r.randn(self.num_outputs, self.hidden_units) / np.sqrt(self.num_inputs)
            self.params["b2"] = np.zeros((self.num_outputs, 1)) 
        else:
            self.params["W"] = np.random.randn(self.hidden_units, self.num_inputs) / np.sqrt(self.num_inputs)
            self.params["b1"] = np.zeros((self.hidden_units, 1))
            self.params["C"] = np.random.randn(self.num_outputs, self.hidden_units) / np.sqrt(self.num_inputs)
            self.params["b2"] = np.zeros((self.num_outputs, 1))
        print("training sample size: [{}]\ntest sample size:[{}]\nhidden units number: [{}]\nbatch_size:[{}]".format(self.x_train.shape, self.x_test.shape, self.hidden_units, self.batch_size))

    def activation(self, z):
        """
        z: must be of size (hidden_units * 1)
        """
        return [*map(lambda x: x if x > 0 else 0, z)]

    def activation_gradient(self, z):
        """
        z: must be of size (hidden_units * 1)
        """
        return [*map(lambda x: 1 if x > 0 else 0, z)]

    def softmax(self, U):
        temp = np.exp(U)
        return temp / np.sum(temp)


    def forward_propagation(self):
        random_index = np.random.choice(self.x_train.shape[0], replace=False, size=self.batch_size)
        self.x_train_sub_samples = self.x_train[random_index].reshape((-1, self.batch_size))
        self.y_train_sub_samples = self.y_train[random_index]
        self.forward_results = {}
        self.forward_results["Z"] = np.dot(self.params["W"], self.x_train_sub_samples) + self.params["b1"]
        self.forward_results["H"] = np.apply_along_axis(self.activation, 0, self.forward_results["Z"])
        self.forward_results["U"] = np.dot(self.params["C"], self.forward_results["H"]) + self.params["b2"]
        self.forward_results["S"] = np.apply_along_axis(self.softmax, 0, self.forward_results["U"])

    def create_unit_matrix(self):
        ey = np.zeros((self.num_outputs, self.batch_size))
        for col_index, row_index in enumerate(self.y_train_sub_samples):
            ey[row_index, col_index] = 1
        return(ey)

    def back_propagation(self):
        ey = self.create_unit_matrix()
        temp = - (ey - self.forward_results["S"])
        self.gradients["db2"] = np.mean(temp, axis=1, keepdims=True)
        self.gradients["dC"] = np.dot(temp, self.forward_results["H"].T) / self.batch_size
        self.gradients["dH"] = np.mean(np.dot(self.params["C"].T, temp), axis=1, keepdims=True)
        H_gradient = np.apply_along_axis(self.activation_gradient, 0, self.forward_results["Z"])
        temp2 = np.multiply(self.gradients["dH"], H_gradient)
        self.gradients["db1"] = np.mean(temp2, axis=1, keepdims=True)
        self.gradients["dW"] = np.dot(temp2, self.x_train_sub_samples.T) / self.batch_size

    def train(self):
        for epoch in range(self.num_epochs):
            if (epoch > 5):
                self.learning_rate = 0.001
            if (epoch > 10):
                self.learning_rate = 0.0001
            if (epoch > 15):
                self.learning_rate = 0.00001
            total_correct = 0
            for i in range(int(self.x_train.shape[0] / self.batch_size)):
                self.forward_propagation()
                prediction_train =  np.argmax(self.forward_results["S"], axis=0)
                total_correct += np.sum(prediction_train == self.y_train_sub_samples)
                self.back_propagation()
                self.params["W"] -= self.learning_rate * self.gradients["dW"]
                self.params["b1"] -= self.learning_rate * self.gradients["db1"]
                self.params["C"] -= self.learning_rate * self.gradients["dC"]
                self.params["b2"] -= self.learning_rate * self.gradients["db2"]
            print("epoch:{} | Training Accuracy:[{}]".format(epoch+1, total_correct/len(self.x_train)))
    def test(self):
        self.Z = np.dot(self.params["W"], self.x_test.T) + self.params["b1"]
        self.H = np.apply_along_axis(self.activation, 0, self.Z)
        self.U = np.dot(self.params["C"], self.H) + self.params["b2"]
        self.S = np.apply_along_axis(self.softmax, 0, self.U)
        self.prediction = np.apply_along_axis(np.argmax, 0, self.S)
        correct_ratio = np.mean(self.prediction == self.y_test)
        return correct_ratio

if __name__ == "__main__":
    nn = MnistModel(x_train, y_train, x_test, 
y_test, hidden_units=100, batch_size=1, learning_rate=0.01, num_epochs=5, seed=1234)
    start = time.time()
    nn.train()
    end = time.time()
    print("Running Time: [{}] second".format(end - start))
    print("Test Accuracy: [{}]".format(nn.test()))