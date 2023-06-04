import torch
import torch.nn as nn
from torch.autograd import grad
from torchvision import utils

from network import *
from functions import *

class Trainer(nn.Module):
    def __init__(self, config):
        super(Trainer, self).__init__()
        self.config = config
        # Generator
        self.genA = generator()
        self.genB = generator()
        # Discriminator
        self.disA = Discriminator()
        # Age Classifier
        self.classifier = VGG()
        # params
        self.gen_params = list(self.genA.parameters()) + list(self.genB.parameters())
        self.dis_params = list(self.disA.parameters())
        self.gen_opt = torch.optim.Adam(self.gen_params, lr=config['lr'], betas=(config['beta_1'], config['beta_2']), weight_decay=config['weight_decay'])
        self.dis_opt = torch.optim.Adam(self.dis_params, lr=config['lr'], betas=(config['beta_1'], config['beta_2']), weight_decay=config['weight_decay'])
        self.gen_scheduler = torch.optim.lr_scheduler.StepLR(self.gen_opt, step_size=config['step_size'], gamma=config['gamma'])
        self.dis_scheduler = torch.optim.lr_scheduler.StepLR(self.dis_opt, step_size=config['step_size'], gamma=config['gamma'])
        
    def initialize(self, vgg_dir):
        self.genA.apply(init_weights)
        self.genB.apply(init_weights)
        self.disA.apply(init_weights)
        vgg_state_dict = torch.load(vgg_dir)
        vgg_state_dict = {k.replace('-', '_'): v for k, v in vgg_state_dict.items()}
        self.classifier.load_state_dict(vgg_state_dict)

    def L1loss(self, input, target):
        return torch.mean(torch.abs(input - target))
    
    def L2loss(self, input, target):
        return torch.mean((input - target)**2)

    def CEloss(self, x, target_age):
        return nn.CrossEntropyLoss()(x, target_age)

    def GAN_loss(self, x, real=True):
        if real:
            target = torch.ones(x.size()).type_as(x)
        else:
            target = torch.zeros(x.size()).type_as(x)
        return nn.MSELoss(reduction='none')(x, target)

    def grad_penalty_r1(self, net, x, coeff=10):
        x.requires_grad=True
        real_predict = net(x)
        gradients = grad(outputs=real_predict.mean(), inputs=x, create_graph=True)[0]
        gradients = gradients.view(gradients.size(0), -1)
        gradient_penalty = (coeff/2) * ((gradients.norm(2, dim=1) ** 2).mean())
        return gradient_penalty
    
    def random_age(self, age_input, diff_val=20):
        age_output = age_input.clone()
        if diff_val > (self.config['age_max'] - self.config['age_min'])/2:
            diff_val = (self.config['age_max'] - self.config['age_min'])//2
        for i, age_ele in enumerate(age_output):
            if age_ele < self.config['age_min'] + diff_val:
                age_target = age_ele.clone().random_(age_ele + diff_val, self.config['age_max'])
            elif (self.config['age_min'] + diff_val) <= age_ele <= (self.config['age_max'] - diff_val):
                age_target = age_ele.clone().random_(self.config['age_min'] + 2*diff_val, self.config['age_max']+1)
                if age_target <= age_ele + diff_val:
                    age_target = age_target - 2*diff_val
            elif age_ele > self.config['age_max'] - diff_val:
                age_target = age_ele.clone().random_(self.config['age_min'], age_ele - diff_val)
            age_output[i] = age_target
        return age_output

    def gen_encode(self, net, x_a, age_a, age_b=0, training=False, target_age=0):
        if target_age:
            self.target_age = target_age
            age_modif = self.target_age*torch.ones(age_a.size()).type_as(age_a)
        else:
            age_modif = self.random_age(age_a, diff_val=25)

        recon_image, modif_image = net(x_a, age_a, age_modif)
        
        return recon_image, modif_image, age_modif
    
    def compute_gen_loss(self, x_a, x_b, age_a, age_b):
        x_a_recon, x_a_modif, age_a_modif = self.gen_encode(self.genA, x_a, age_a, age_b, training=True)
        x_b_recon, x_b_modif, age_b_modif = self.gen_encode(self.genB, x_b, age_b, age_a, training=True)

        _, x_b_cycle, _ = self.gen_encode(self.genA, x_b_modif, age_b_modif, age_a, training=True, target_age= int(age_b.cpu().numpy()[0]))
        _, x_a_cycle, _ = self.gen_encode(self.genB, x_a_modif, age_a_modif, age_b, training=True, target_age= int(age_a.cpu().numpy()[0]))

        realism_a_modif = self.disA(x_a_modif)
        realism_b_modif = self.disA(x_b_modif)
        predict_age_a_pb = self.classifier(vgg_transform(x_a_modif))['fc8']
        predict_age_b_pb = self.classifier(vgg_transform(x_b_modif))['fc8']

        self.loss_class_a = self.CEloss(predict_age_a_pb, age_a_modif)  
        self.loss_class_b = self.CEloss(predict_age_b_pb, age_b_modif)
        self.loss_recon_a = self.L1loss(x_a_recon, x_a)  
        self.loss_recon_b = self.L1loss(x_b_recon, x_b) 
        self.loss_cycle_a = self.L1loss(x_a_cycle, x_a)
        self.loss_cycle_b = self.L1loss(x_b_cycle, x_b)
        self.loss_adver_a = self.GAN_loss(realism_a_modif, True).mean()
        self.loss_adver_b = self.GAN_loss(realism_b_modif, True).mean()

        self.loss_gen = self.config['w']['recon']*self.loss_recon_a + self.config['w']['class']*self.loss_class_a + \
                        self.config['w']['cycle']*self.loss_cycle_a + self.config['w']['adver']*self.loss_adver_a + \
                        self.config['w']['recon']*self.loss_recon_b + self.config['w']['class']*self.loss_class_b + \
                        self.config['w']['cycle']*self.loss_cycle_b + self.config['w']['adver']*self.loss_adver_b\

                        
        return self.loss_gen
    
    def compute_dis_loss(self, x_a, x_b, age_a, age_b):
        # Generate modified image
        _, x_a_modif, _ = self.gen_encode(self.genA, x_a, age_a, age_b, training=True)

        self.realism_a_modif = self.disA(x_a_modif.detach())
        self.realism_b = self.disA(x_b)

        self.loss_gp_a = self.grad_penalty_r1(self.disA, x_b)  
        self.loss_dis_a = self.GAN_loss(self.realism_b, True).mean() + self.GAN_loss(self.realism_a_modif, False).mean()
        self.loss_dis_gp = self.config['w']['dis']*self.loss_dis_a + self.config['w']['gp']*self.loss_gp_a
        return self.loss_dis_gp

    
    def save_image(self, x_a, age_a, log_dir, n_epoch, n_iter):
        x_a_recon, x_a_modif, age_a_modif = self.gen_encode(self.genA, x_a, age_a)
        utils.save_image(clip_img(x_a), log_dir + 'epoch' +str(n_epoch+1)+ 'iter' +str(n_iter+1)+ '_content.png')
        utils.save_image(clip_img(x_a_recon), log_dir + 'epoch' +str(n_epoch+1)+ 'iter' +str(n_iter+1)+ '_content_recon_'+str(age_a.cpu().numpy()[0])+'.png')
        utils.save_image(clip_img(x_a_modif), log_dir + 'epoch' +str(n_epoch+1)+ 'iter' +str(n_iter+1)+ '_content_modif_'+str(age_a_modif.cpu().numpy()[0])+'.png')
    
    def test_eval(self, x_a, age_a, target_age=0, hist_trans=True):
        _, x_a_modif, _= self.gen_encode(self.genA, x_a, age_a, target_age=target_age)
        if hist_trans:
            for j in range(x_a_modif.size(0)):
                x_a_modif[j] = hist_transform(x_a_modif[j], x_a[j])
        return x_a_modif
    
    def save_model(self, log_dir):
        torch.save(self.genA.state_dict(),'{:s}/genA.pth.tar'.format(log_dir))
        torch.save(self.genB.state_dict(),'{:s}/genA.pth.tar'.format(log_dir))
        torch.save(self.disA.state_dict(),'{:s}/disA.pth.tar'.format(log_dir))

    def save_checkpoint(self, n_epoch, log_dir):
        checkpoint_state = {
            'n_epoch': n_epoch,
            'genA_state_dict': self.genA.state_dict(),
            'genB_state_dict': self.genB.state_dict(),
            'disA_state_dict': self.disA.state_dict(),
            'gen_opt_state_dict': self.gen_opt.state_dict(),
            'dis_opt_state_dict': self.dis_opt.state_dict(),
            'gen_scheduler_state_dict': self.gen_scheduler.state_dict(),
            'dis_scheduler_state_dict': self.dis_scheduler.state_dict()
        } 
        torch.save(checkpoint_state, '{:s}/checkpoint'.format(log_dir))
        if (n_epoch+1) % 10 == 0 :
            torch.save(checkpoint_state, '{:s}/checkpoint'.format(log_dir)+'_'+str(n_epoch+1))
    
    def load_model(self, log_dir):
        self.genA.load_state_dict(torch.load('{:s}/genA.pth.tar'.format(log_dir)))
        self.genB.load_state_dict(torch.load('{:s}/genB.pth.tar'.format(log_dir)))
        self.disA.load_state_dict(torch.load('{:s}/disA.pth.tar'.format(log_dir)))

    def load_checkpoint(self, checkpoint_path):
        state_dict = torch.load(checkpoint_path)
        self.genA.load_state_dict(state_dict['genA_state_dict'])
        self.genB.load_state_dict(state_dict['genB_state_dict'])
        self.disA.load_state_dict(state_dict['disA_state_dict'])
        self.gen_opt.load_state_dict(state_dict['gen_opt_state_dict'])
        self.dis_opt.load_state_dict(state_dict['dis_opt_state_dict'])
        self.gen_scheduler.load_state_dict(state_dict['gen_scheduler_state_dict'])
        self.dis_scheduler.load_state_dict(state_dict['dis_scheduler_state_dict'])
        return state_dict['n_epoch'] + 1

    def update(self, x_a, x_b, age_a, age_b, n_iter):
        self.n_iter = n_iter
        self.dis_opt.zero_grad()
        self.compute_dis_loss(x_a, x_b, age_a, age_b).backward()
        self.dis_opt.step()
        self.gen_opt.zero_grad()
        self.compute_gen_loss(x_a, x_b, age_a, age_b).backward()
        self.gen_opt.step()

