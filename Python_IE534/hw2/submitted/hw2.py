import numpy as np
import h5py
import time 
import copy
from scipy import signal
file_name = "../data/MNISTdata.hdf5"
data = h5py.File(file_name, "r")
x_train = np.float32(data["x_train"][:]).reshape(-1, 28, 28)
y_train = np.int32(np.hstack(np.array(data["y_train"]))).reshape(-1,1)
x_test = np.float32(data["x_test"][:]).reshape(-1, 28, 28)
y_test = np.int32(np.hstack(np.array(data["y_test"]))).reshape(-1,1)
data.close()
def Convolution(image, myfilter):
    d = image.shape[-1]
    ky, kx = myfilter.shape
    conv = np.zeros((d - ky + 1, d - kx + 1))
    for i in range(d - ky + 1):
        for j in range(d - kx + 1):
            conv[i,j] = np.sum(np.multiply(image[i:i+ky,j:j+kx], myfilter))
    return conv

class CNN():
    def __init__(self, x_train, y_train, x_test, y_test, num_channels=5, learning_rate=0.01, num_epochs=5):
        self.x_train = x_train
        self.x_test = x_test
        self.y_train = y_train
        self.y_test = y_test
        
        self.num_outputs = 10
        self.learning_rate = learning_rate
        self.num_epochs = num_epochs
        self.num_channels = num_channels
        self.d = self.x_train.shape[1]
        
        self.params = {}
        r = np.random.RandomState(1234)
        self.params["K"] = r.randn(5, 5, self.num_channels) / self.d
        self.ky = self.params["K"].shape[0]
        self.kx = self.params["K"].shape[1]
        
        self.params["W"] = r.randn(self.num_outputs, 
                                   self.d - self.ky + 1,
                                   self.d - self.kx + 1,
                                   self.num_channels) / self.d
        self.params["b"] = np.zeros((self.num_outputs, 1))
        
        self.gradients = {}
        
        print("training sample size: [{}]\ntest sample size:[{}]\nchannels:[{}]".format(self.x_train.shape, self.x_test.shape, self.num_channels))


    def convolution_process(self, img):
        convoluted = np.zeros((self.d - self.ky + 1, 
                               self.d - self.kx + 1,
                               self.params["K"].shape[2]))
        for filter_idx in range(self.params["K"].shape[2]):
            convoluted[:, :, filter_idx] = Convolution(img[0,:,:], self.params["K"][ :, :, filter_idx])
            #signal.correlate2d(img[0,:,:], self.params["K"][ :, :, filter_idx], mode='valid', boundary='wrap')
        return convoluted
        
    def relu(self, Z):
        U = copy.deepcopy(Z)
        U[U<=0] = 0
        return  U

    def relu_gradient(self, Z):
        dZ = copy.deepcopy(Z)
        dZ[dZ >= 0] = 1
        dZ[dZ < 0] = 0
        return  dZ

    def softmax(self, U):
        temp = np.exp(U)
        return temp / np.sum(temp)

    def forward_propagation(self):
        random_index = np.random.randint(self.x_train.shape[0])
        self.img = self.x_train[random_index].reshape((1, self.d, self.d))
        self.img_label = self.y_train[random_index].reshape((-1,1))
        self.forward_results = {}
        self.forward_results["Z"] = self.convolution_process(self.img)
        self.forward_results["H"] = self.relu(self.forward_results["Z"])
        self.forward_results["U"] = np.tensordot(self.params["W"],
                                                 self.forward_results["H"], 
                                                 axes=3).reshape((self.num_outputs ,1)) + self.params["b"]
        self.forward_results["S"] = self.softmax(self.forward_results["U"])

    def back_propagation(self):
        ey = np.zeros((self.num_outputs, 1)); ey[self.img_label] = 1
        self.gradients["dU"] = - (ey - self.forward_results["S"])
        self.gradients["db"] = self.gradients["dU"]
        self.gradients["delta"] = np.tensordot(self.gradients["dU"].squeeze(), self.params["W"], axes=1)
        self.gradients["dW"] = np.tensordot(self.gradients["dU"].squeeze(), self.forward_results["H"], axes=0)
        dsigmaZ = self.relu_gradient(self.forward_results["Z"])
        temp = np.multiply(dsigmaZ, self.gradients["delta"])
        self.gradients["dK"] = copy.deepcopy(self.params["K"])
        for filter_idx in range(self.params["K"].shape[2]):
            self.gradients["dK"][:,:,filter_idx] = Convolution(self.img[0,:,:], temp[:,:,filter_idx])
            # signal.correlate2d(self.img[0,:,:], temp[:,:,filter_idx], mode='valid',  boundary='wrap')                                         
    def train(self):
        for epoch in range(self.num_epochs):
            if (epoch > 5):
                self.learning_rate = 0.001
            if (epoch > 10):
                self.learning_rate = 0.0001
            if (epoch > 15):
                self.learning_rate = 0.00001
            total_correct = 0
            for i in range(int(self.x_train.shape[0])):
                if ( i % 10000 == 0):
                    print(i)
                self.forward_propagation()
                prediction_train =  np.argmax(self.forward_results["S"], axis=0)
                total_correct += np.sum(prediction_train == self.img_label)
                self.back_propagation()
                self.params["b"] -= self.learning_rate * self.gradients["db"]
                self.params["W"] -= self.learning_rate * self.gradients["dW"]
                self.params["K"] -= self.learning_rate * self.gradients["dK"]
            print("epoch:{} | Training Accuracy:[{}]".format(epoch+1, total_correct/(self.x_train.shape[0])))
    def test(self):
        total_correct_test = 0
        for img, img_label in zip(self.x_test, self.y_test):
            img = img.reshape((1, self.d, self.d))
            img_label = img_label.reshape((-1,1))
            Z = self.convolution_process(img)
            H = self.relu(Z)
            U = np.tensordot(self.params["W"], H, axes=3).reshape((self.num_outputs ,1)) + self.params["b"]
            S = self.softmax(U)
            prediction_test = np.argmax(S, axis=0)
            total_correct_test += np.sum(prediction_test == img_label)
        correct_ratio = total_correct_test / self.x_test.shape[0]
        return correct_ratio

if __name__ == "__main__":
    myCNN = CNN(x_train, y_train, x_test, y_test,  num_channels=5, learning_rate=0.01, num_epochs=5)
    start = time.time()
    myCNN.train()
    end = time.time()
    print("Total time used for training with 5 epochs is [{}] seconds".format(end - start))
    myCNN.test()