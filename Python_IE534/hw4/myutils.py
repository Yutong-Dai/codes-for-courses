import torch
import shutil


def save_checkpoint(state, is_best, filename='checkpoint.pth.tar', extra=""):
    torch.save(state, extra+filename)
    if is_best:
        shutil.copyfile(extra+filename, extra+'model_best.pth.tar')


def update_learning_rate(optimizer, lr):
    for param_group in optimizer.param_groups:
        param_group['lr'] = lr
