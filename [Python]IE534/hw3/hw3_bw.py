import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torch.autograd import Variable
from torchvision import transforms
import h5py
import copy
import time
file_name = "../data/CIFAR10.hdf5"
data = h5py.File(file_name, "r")
x_train = np.float32(data["X_train"][:]).reshape(-1, 3, 32, 32)
y_train = np.int32(np.array(data["Y_train"]))
x_test = np.float32(data["X_test"][:]).reshape(-1, 3, 32, 32)
y_test = np.int32(np.array(data["Y_test"]))
data.close()

def random_flip(x_train,y_train,portion=0.5, direction=2):
    """
    portion:
        portion of sample in x_train get flipped
    direction:
        2 - horizontal
        1 - vertical
    """
    all_index = x_train.shape[0]
    idx = np.random.choice(all_index, np.int(portion*all_index), replace=False)
    new = copy.deepcopy(x_train)
    new[idx,[0],:,:] = np.flip(new[idx,[0],:,:],direction)
    new[idx,[1],:,:] = np.flip(new[idx,[1],:,:],direction)
    new[idx,[2],:,:] = np.flip(new[idx,[2],:,:],direction)
    return y_train, new

class CNN_torch(nn.Module):
    def __init__(self):
        super(CNN_torch, self).__init__()
        self.conv1 = nn.Conv2d(in_channels=3, out_channels=64, kernel_size=4, stride=1, padding=2, bias=True)
        self.batch_norm1 = nn.BatchNorm2d(num_features=64, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True)
        self.conv2 = nn.Conv2d(in_channels=64, out_channels=64, kernel_size=4, stride=1, padding=2, bias=True)
        self.conv3 = nn.Conv2d(in_channels=64, out_channels=64, kernel_size=4, stride=1, padding=2, bias=True)
        self.batch_norm2 = nn.BatchNorm2d(num_features=64, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True)
        self.conv4 = nn.Conv2d(in_channels=64, out_channels=64, kernel_size=4, stride=1, padding=2, bias=True)
        self.conv5 = nn.Conv2d(in_channels=64, out_channels=64, kernel_size=4, stride=1, padding=2, bias=True)
        self.batch_norm3 = nn.BatchNorm2d(num_features=64, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True)
        self.conv6 = nn.Conv2d(in_channels=64, out_channels=64, kernel_size=3, stride=1, padding=0, bias=True)
        self.conv7 = nn.Conv2d(in_channels=64, out_channels=64, kernel_size=3, stride=1, padding=0, bias=True)
        self.batch_norm4 = nn.BatchNorm2d(num_features=64, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True)
        self.conv8 = nn.Conv2d(in_channels=64, out_channels=64, kernel_size=3, stride=1, padding=0, bias=True)
        self.batch_norm5 = nn.BatchNorm2d(num_features=64, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True)
        self.fc1 = nn.Linear(64 * 4 * 4, 500)
        self.fc2 = nn.Linear(500, 10)
    def forward(self, x):
        #print("con0: x shape{}".format(x.shape))
        x = F.relu(self.conv1(x))
        #print("con1: x shape{}".format(x.shape))
        x = self.batch_norm1(x)
        x = F.relu(self.conv2(x))
        #print("con2: x shape{}".format(x.shape))
        x = F.dropout2d(F.max_pool2d(x, kernel_size=2, stride=2), p=0.25)
        #print("maxpool1: x shape{}".format(x.shape))
        x = F.relu(self.conv3(x))
        #print("con3: x shape{}".format(x.shape))
        x = self.batch_norm2(x)
        x = F.relu(self.conv4(x))
        #print("con4: x shape{}".format(x.shape))
        x = F.dropout2d(F.max_pool2d(x, kernel_size=2, stride=2), p=0.25)
        #print("maxpool2: x shape{}".format(x.shape))
        x = F.relu(self.conv5(x))
        #print("con5: x shape{}".format(x.shape))
        x = self.batch_norm3(x)
        x = F.relu(self.conv6(x))
        #print("con6: x shape{}".format(x.shape))
        x = F.dropout2d(x, p=0.25)
        x = F.relu(self.conv7(x))
        #print("con7: x shape{}".format(x.shape))
        x = self.batch_norm4(x)
        x = F.relu(self.conv8(x))
        #print("con8: x shape{}".format(x.shape))
        x = self.batch_norm5(x)
        x = F.dropout2d(x, p=0.25)
        #print("final: x shape{}".format(x.shape))
        x = x.view(-1, self.fc1.in_features)
        #print("flatten: x shape{}".format(x.shape))
        x = self.fc1(x)
        x = self.fc2(x)
        return F.log_softmax(x, dim=1)
    
model = CNN_torch()
use_cuda = torch.cuda.is_available()
if use_cuda:
    model.cuda()
    model = torch.nn.DataParallel(model, device_ids=range(torch.cuda.device_count()))
    cudnn.benchmark = True
criterion = nn.CrossEntropyLoss()
optimizer = optim.Adam(model.parameters())
batch_size = 100
num_epoch = 30
train_loss = []; train_accuracy_epcoh = []
L_Y_train = len(y_train)
L_Y_test = len(y_test)

model.train()
for epoch in range(num_epoch):
    index_permutation = np.random.permutation(L_Y_train)
    x_train = x_train[index_permutation, :]
    y_train = y_train[index_permutation]
    if np.random.uniform(0,1) < 0.5:
        y_train_aug, x_train_aug = random_flip(x_train, y_train, portion=0.6, direction=np.random.randint(1,3))
    else:
        y_train_aug, x_train_aug = y_train, x_train
    x_train_aug = torch.from_numpy(x_train_aug)
    train_accuracy = []
    for i in range(0, L_Y_train, batch_size):
        x_train_batch = torch.FloatTensor(x_train_aug[i:i+batch_size, :])
        y_train_batch = torch.LongTensor(y_train_aug[i:i+batch_size])
        if use_cuda:
            data, target = Variable(x_train_batch).cuda(), Variable(y_train_batch).cuda()
        else:
            data, target = Variable(x_train_batch), Variable(y_train_batch)
        optimizer.zero_grad()
        output = model(data)
        loss = F.nll_loss(output, target)
        loss.backward()
        if torch.__version__ == '0.4.1':
            train_loss.append(loss.item())
        else:
            train_loss.append(loss.data[0])
        # update parameters
        optimizer.step()
        prediction = output.data.max(1)[1]
        accuracy = (float(prediction.eq(target.data).sum()) / float(batch_size))
        train_accuracy.append(accuracy)
    train_accuracy_epcoh.append(np.mean(train_accuracy)) 
    print("Epoch: {} | Loss: {} | Accuracy::{}".format(epoch+1, train_loss[-1], train_accuracy_epcoh[-1]))