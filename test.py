import argparse
import os
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.utils.data as data
import yaml

from PIL import Image
from torchvision import transforms, utils

from network import *
from functions import *
from trainer import *

torch.backends.cudnn.enabled = True
torch.backends.cudnn.deterministic = True
torch.backends.cudnn.benchmark = True
Image.MAX_IMAGE_PIXELS = None
if torch.cuda.is_available():
    device = torch.device('cuda')
else:
    device = torch.device('cpu')

parser = argparse.ArgumentParser()
parser.add_argument('--config', type=str, default='params', help='path to the config file.')
parser.add_argument('--vgg_model_path', type=str, default='./models/dex_imdb_wiki.caffemodel.pt', help='pretrained age classifier')
parser.add_argument('--log_path', type=str, default='./logs/', help='log file path')
parser.add_argument('--checkpoint', type=str, default='', help='checkpoint file path')
parser.add_argument('--img_path', type=str, default='./test/input/', help='test image path')
parser.add_argument('--out_path', type=str, default='./test/output/', help='test output path')
parser.add_argument('--target_age', type=int, default=65, help='Age transform target, interger value between 20 and 70')
opts = parser.parse_args()

log_dir = os.path.join(opts.log_path, opts.config) + '/'
if not os.path.exists(opts.out_path):
    os.makedirs(opts.out_path)

config = yaml.full_load(open('./configs/' + opts.config + '.yaml', 'r'))
img_size = (config['input_w'], config['input_h'])

# Initialize trainer
trainer = Trainer(config)

# Load pretrained model 
if opts.checkpoint:
    trainer.load_checkpoint(opts.checkpoint)
else:
    trainer.load_checkpoint(log_dir + 'checkpoint')

trainer.to(device)

# Load test image
img_list = os.listdir(opts.img_path)
img_list.sort()

# Preprocess
def preprocess(img_name):
    resize = transforms.Compose([
            transforms.Resize(img_size),
            transforms.ToTensor()
            ])
    normalize = transforms.Normalize(mean=[0.48501961, 0.45795686, 0.40760392], std=[1,1,1])
    img_pil = Image.open(opts.img_path + img_name)
    img = resize(img_pil)
    if img.size(0) == 1:
        img = torch.cat((img, img, img), dim = 0)
    img = normalize(img)
    return img

# Set target age
target_age = opts.target_age
predicted_age = []
dis_loss = []

# update age classifier weight
vgg_state_dict = torch.load(opts.vgg_model_path)
vgg_state_dict = {k.replace('-', '_'): v for k, v in vgg_state_dict.items()}
trainer.classifier.load_state_dict(vgg_state_dict)
# update disicriminator weight
state_dict = torch.load(opts.checkpoint)
trainer.disA.load_state_dict(state_dict['dis_state_dict'])

with torch.no_grad():
    for img_name in img_list:
        if not img_name.endswith(('png', 'jpg', 'PNG', 'JPG')):
            print('File ignored: ' + img_name)
            continue
        image_A = preprocess(img_name)
        image_A = image_A.unsqueeze(0).to(device)

        age_modif = torch.tensor(target_age).unsqueeze(0).to(device)
        image_A_modif = trainer.test_eval(image_A, age_modif, target_age=target_age, hist_trans=True)  
        
        predict_age_pb = trainer.classifier(vgg_transform(image_A_modif))['fc8']
        predict_age = get_predict_age(predict_age_pb)
        predicted_age.append(predict_age)
        realism_b = trainer.disA(image_A)
        realism_a_modif = trainer.disA(image_A_modif)
        loss_dis = trainer.GAN_loss(realism_b, True).mean() + trainer.GAN_loss(realism_a_modif, False).mean()
        dis_loss.append(loss_dis)
        utils.save_image(clip_img(image_A_modif), opts.out_path + img_name.split('.')[0] + '_age_' + str(target_age) + '.jpg')
    print(sum(predicted_age)/len(predicted_age))
    print(sum(dis_loss)/ len(dis_loss))

