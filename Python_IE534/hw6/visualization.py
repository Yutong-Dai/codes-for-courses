'''
File: visualization.py
Author: Yutong Dai (rothdyt@gmail.com)
File Created: Friday, 2018-10-28 09:07
Last Modified: Sunday, 2018-10-28 15:01
--------------------------------------------
Desscription: 
'''
import torch
import torchvision
import torchvision.transforms as transforms
import numpy as np
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torch.autograd import Variable
from torch import autograd
import utils
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
import os
import time
import utils

batch_size = 100
transform_test = transforms.Compose([
    transforms.CenterCrop(32),
    transforms.ToTensor(),
    transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5)),
])

testset = torchvision.datasets.CIFAR10(root='./', train=False, download=False, transform=transform_test)
testloader = torch.utils.data.DataLoader(testset, batch_size=batch_size, shuffle=False, num_workers=8)
testloader = enumerate(testloader)

"""
part 1
"""
model = torch.load('cifar10.model')
model.cuda()
model.eval()

batch_idx, (X_batch, Y_batch) = testloader.__next__()
X_batch = Variable(X_batch, requires_grad=True).cuda()
Y_batch_alternate = (Y_batch + 1) % 10
Y_batch_alternate = Variable(Y_batch_alternate).cuda()
Y_batch = Variable(Y_batch).cuda()
# save real images
samples = X_batch.data.cpu().numpy()
samples += 1.0
samples /= 2.0
samples = samples.transpose(0, 2, 3, 1)
fig = utils.plot(samples[0:100])
plt.savefig('./visualization/real_images.png', bbox_inches='tight')
plt.close(fig)

output = model(X_batch)[1]
prediction = output.data.max(1)[1]  # first column has actual prob.
accuracy = (float(prediction.eq(Y_batch.data).sum()) / float(batch_size))*100.0
print(accuracy)
# slightly jitter all input images
criterion = nn.CrossEntropyLoss(reduce=False)
loss = criterion(output, Y_batch_alternate)
gradients = torch.autograd.grad(outputs=loss, inputs=X_batch,
                                grad_outputs=torch.ones(loss.size()).cuda(),
                                create_graph=True, retain_graph=False, only_inputs=True)[0]

gradient_image = gradients.data.cpu().numpy()
gradient_image = (gradient_image - np.min(gradient_image))/(np.max(gradient_image)-np.min(gradient_image))
gradient_image = gradient_image.transpose(0, 2, 3, 1)
fig = utils.plot(gradient_image[0:100])
plt.savefig('./visualization/gradient_image.png', bbox_inches='tight')
plt.close(fig)
# jitter input image
gradients[gradients > 0.0] = 1.0
gradients[gradients < 0.0] = -1.0
gain = 8.0
X_batch_modified = X_batch - gain*0.007843137*gradients
X_batch_modified[X_batch_modified > 1.0] = 1.0
X_batch_modified[X_batch_modified < -1.0] = -1.0

# evaluate new fake images
output = model(X_batch_modified)
output = output[1]
prediction = output.data.max(1)[1]  # first column has actual prob.
accuracy = (float(prediction.eq(Y_batch.data).sum()) / float(batch_size))*100.0
print(accuracy)

# save fake images
samples = X_batch_modified.data.cpu().numpy()
samples += 1.0
samples /= 2.0
samples = samples.transpose(0, 2, 3, 1)
fig = utils.plot(samples[0:100])
plt.savefig('./visualization/jittered_images.png', bbox_inches='tight')
plt.close(fig)


"""
Part2:
    plots for the discriminator without the generator:
        Synthetic Images Maximizing Classification Output
"""
model = torch.load('cifar10.model')
model.cuda()
model.eval()

utils.plot_max_class(X_batch, model, label="without")
utils.plot_max_feature(X_batch, model, extract_features=2, label="without", batch_size=100)
utils.plot_max_feature(X_batch, model, extract_features=4, label="without", batch_size=100)
utils.plot_max_feature(X_batch, model, extract_features=8, label="without", batch_size=100)


"""
Part3:
    plots for the discriminator with the generator:
        Synthetic Images Maximizing Classification Output
"""
model = torch.load('discriminator.model')
model.cuda()
model.eval()

utils.plot_max_class(X_batch, model, label="with")
utils.plot_max_feature(X_batch, model, extract_features=2, label="with", batch_size=100)
utils.plot_max_feature(X_batch, model, extract_features=4, label="with", batch_size=100)
utils.plot_max_feature(X_batch, model, extract_features=8, label="with", batch_size=100)
