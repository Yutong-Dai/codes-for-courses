import torchvision
import torchvision.transforms as transforms
import torch
from utils import generate_training_data_set_for_current_epoch, TinyImageNet
from torch.utils.data.sampler import SubsetRandomSampler


transform_train = transforms.Compose([
    transforms.RandomCrop(64, padding=4),
    transforms.RandomHorizontalFlip(),
    transforms.ToTensor(),
])

img_triplet, label_triplet = generate_training_data_set_for_current_epoch()
train = TinyImageNet(img_triplet, label_triplet, transform=transform_train)
train_loader = torch.utils.data.DataLoader(train, batch_size=2,
                                           shuffle=False, sampler=SubsetRandomSampler(range(10)))

for idx, (data, target) in enumerate(train_loader):
    print(idx, data[0].shape, target[2])
