import torch
import torchvision
import torchvision.transforms as transforms
import numpy as np
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torch.autograd import Variable
import utils

transform_train = transforms.Compose([
    transforms.RandomResizedCrop(32, scale=(0.7, 1.0), ratio=(1.0, 1.0)),
    transforms.ColorJitter(
        brightness=0.1*torch.randn(1),
        contrast=0.1*torch.randn(1),
        saturation=0.1*torch.randn(1),
        hue=0.1*torch.randn(1)),
    transforms.RandomHorizontalFlip(),
    transforms.ToTensor(),
    transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5)),
])

transform_test = transforms.Compose([
    transforms.CenterCrop(32),
    transforms.ToTensor(),
    transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5)),
])


batch_size = 128
learning_rate = 0.0001
num_epochs = 100

trainset = torchvision.datasets.CIFAR10(root='./', train=True, download=True, transform=transform_train)
trainloader = torch.utils.data.DataLoader(trainset, batch_size=batch_size, shuffle=True, num_workers=8)

testset = torchvision.datasets.CIFAR10(root='./', train=False, download=False, transform=transform_test)
testloader = torch.utils.data.DataLoader(testset, batch_size=batch_size, shuffle=False, num_workers=8)

model = utils.discriminator()
model.cuda()
criterion = nn.CrossEntropyLoss()
optimizer = torch.optim.Adam(model.parameters(), lr=learning_rate)
for epoch in range(num_epochs):
    for group in optimizer.param_groups:
        for p in group['params']:
            state = optimizer.state[p]
            if('step' in state and state['step'] >= 1024):
                state['step'] = 1000
    if(epoch == 50):
        for param_group in optimizer.param_groups:
            param_group['lr'] = learning_rate/10.0
    if(epoch == 75):
        for param_group in optimizer.param_groups:
            param_group['lr'] = learning_rate/100.0
    correct = 0
    total = 0
    train_accu = []
    model.train()
    for batch_idx, (X_train_batch, Y_train_batch) in enumerate(trainloader):

        if(Y_train_batch.shape[0] < batch_size):
            continue
        X_train_batch = Variable(X_train_batch).cuda()
        Y_train_batch = Variable(Y_train_batch).cuda()
        output = model(X_train_batch)
        output = output[1]  # batch_size*10
        pred = torch.argmax(output, 1)
        correct += (pred == Y_train_batch.data).sum()
        total += Y_train_batch.data.size(0)
        loss = criterion(output, Y_train_batch)
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
    train_accu.append(float(correct)/float(total))
    X_train_batch.detach()
    Y_train_batch.detach()
    print('train => epoch[{}|{}]: accuracy is {}'.format(epoch+1, num_epochs, float(correct)/float(total)))
    test_accu = []
    model.eval()
    with torch.no_grad():
        correct = 0
        total = 0
        for batch_idx, (X_test_batch, Y_test_batch) in enumerate(testloader):
            if(Y_test_batch.shape[0] < batch_size):
                continue
            X_test_batch = Variable(X_test_batch).cuda()
            Y_test_batch = Variable(Y_test_batch).cuda()
            output = model(X_test_batch)
            output = output[1]
            pred = torch.argmax(output, 1)
            correct += (pred == Y_test_batch.data).sum()
            total += Y_test_batch.data.size(0)
        test_accu.append(float(correct)/float(total))
    print('test => epoch[{}|{}]: accuracy is {}'.format(epoch+1, num_epochs, float(correct)/float(total)))
print('train accuracy is {}'.format(np.mean(train_accu)))
print('test accuracy is {}'.format(np.mean(test_accu)))
torch.save(model, 'cifar10.model')
