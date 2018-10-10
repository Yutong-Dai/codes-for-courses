import torchvision
import torchvision.transforms as transforms
import torch
from utils import TinyImageNet
from torch.utils.data.sampler import SubsetRandomSampler
import h5py


hf = h5py.File('data_lzf.h5', 'r')
img_triplet = hf["img_triplet"][:]
label_triplet = hf['label_triplet'][:]
hf.close()
transform_train = transforms.Compose([
    transforms.RandomCrop(64, padding=4),
    transforms.RandomHorizontalFlip(),
    transforms.ToTensor(),
])
train = TinyImageNet(img_triplet, label_triplet, transform=transform_train)
# train_loader = torch.utils.data.DataLoader(train, batch_size=2,
#                                            shuffle=False, sampler=SubsetRandomSampler(range(10)))
