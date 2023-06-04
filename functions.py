import os
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.utils.data as data
from PIL import Image
from torch.autograd import grad
from torchvision import transforms
        
def clip_img(x):
    img_tmp = x.clone()[0]
    img_tmp[0] += 0.48501961
    img_tmp[1] += 0.45795686
    img_tmp[2] += 0.40760392
    img_tmp = torch.clamp(img_tmp, 0, 1)
    return [img_tmp.detach().cpu()]
    
def hist_transform(source_tensor, target_tensor):
    c, h, w = source_tensor.size()
    s_t = source_tensor.view(c, -1)
    t_t = target_tensor.view(c, -1)
    s_t_sorted, s_t_indices = torch.sort(s_t)
    t_t_sorted, t_t_indices = torch.sort(t_t)
    for i in range(c):
        s_t[i, s_t_indices[i]] = t_t_sorted[i]
    return s_t.view(c, h, w)

def init_weights(m):
    if type(m) == nn.Conv2d:
        nn.init.xavier_uniform_(m.weight)
        nn.init.constant_(m.bias, 0.01)
    elif type(m) == nn.Linear:
        nn.init.uniform_(m.weight, 0.0, 1.0)
        if m.bias is not None:
            nn.init.constant_(m.bias, 0.01)

def reg_loss(img):
    reg_loss = torch.mean(torch.abs(img[:, :, :, :-1] - img[:, :, :, 1:]))\
             + torch.mean(torch.abs(img[:, :, :-1, :] - img[:, :, 1:, :]))
    return reg_loss

def vgg_transform(x):
    r, g, b = torch.split(x, 1, 1)
    out = torch.cat((b, g, r), dim = 1)
    out = F.interpolate(out, size=(224, 224), mode='bilinear')
    out = out*255.
    return out

def get_predict_age(age_pb):
    predict_age_pb = F.softmax(age_pb)
    predict_age = torch.zeros(age_pb.size(0)).type_as(predict_age_pb)
    for i in range(age_pb.size(0)):
        for j in range(age_pb.size(1)):
            predict_age[i] += j*predict_age_pb[i][j]
    return predict_age

# Define Custom Dataset
class MyDataSet(data.Dataset):
    def __init__(self, age_min, age_max, image_dir, label_dir, output_size=(512, 512), training_set=True, obscure_age=True):
        self.image_dir = image_dir
        self.transform = transforms.Normalize(mean=[0.48501961, 0.45795686, 0.40760392], std=[1, 1, 1])
        self.resize = transforms.Compose([
            transforms.Resize(output_size),
            transforms.ToTensor(),
            transforms.RandomHorizontalFlip()
        ])
        label = np.load(label_dir, allow_pickle=True)
        train_len = int(0.95*len(label))
        self.training_set = training_set
        self.obscure_age = obscure_age
        if training_set:
            label = label[:train_len]
        else:
            label = label[train_len:]
        a_mask = np.zeros(len(label), dtype=bool)
        for i in range(len(label)):
            if int(label[i, 1]) in range(age_min, age_max): a_mask[i] = True
        self.label = label[a_mask]
        self.length = len(self.label)
    
    def __len__(self):
        return self.length

    def __getitem__(self, index):
        img_name = os.path.join(self.image_dir, self.label[index][0])
        if self.training_set and self.obscure_age:
            age_val = int(self.label[index][1]) + np.random.randint(-1, 1)
        else:
            age_val = int(self.label[index][1])
        age = torch.tensor(age_val)
        image = Image.open(img_name)
        img = self.resize(image)
        if img.size(0) == 1:
            img = torch.cat((img, img, img), dim = 0)
        img = self.transform(img)
        return img, age
