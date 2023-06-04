import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.nn.utils import spectral_norm

class Conv2d(nn.Module):
    def __init__(self, input_size, output_size, kernel_size, stride, conv='conv', pad='mirror', norm='in', activ='relu', sn=False):
        super(Conv2d, self).__init__()
        # padding
        if pad == 'mirror':
            self.padding = nn.ReflectionPad2d(kernel_size//2)
        elif pad == 'none':
            self.padding = None
        else:
            self.padding = nn.ReflectionPad2d(pad)
        # convolution
        if conv=='conv':
            self.conv = nn.Conv2d(input_size, output_size, kernel_size=kernel_size, stride=stride)
        # Normalization
        if norm == 'in':
            self.norm = nn.InstanceNorm2d(output_size, affine=True)
        elif norm == 'batch':
            self.norm = nn.BatchNorm2d(output_size)
        elif norm == 'none':
            self.norm = None 
        # Activation func.
        if activ == 'relu':
            self.activ = nn.ReLU()
        elif activ == 'leakyrelu':
            self.activ = nn.LeakyReLU(0.2)
        elif activ == 'none':
            self.activ = None
        if sn == True:
            self.conv = spectral_norm(self.conv)
  
    def forward(self, x):
        if self.padding:
            out = self.padding(x)
        else:
            out = x
        out = self.conv(out)
        if self.norm:
            out = self.norm(out)
        if self.activ:
            out = self.activ(out)
        return out
    
class ResBlock(nn.Module):
    def __init__(self, input_size, kernel_size, stride, conv='conv', pad='mirror', norm='in', activ='relu', sn=False):
        super(ResBlock, self).__init__()
        self.block = nn.Sequential(
                Conv2d(input_size, input_size, kernel_size=kernel_size, stride=stride, conv=conv, pad=pad, norm=norm, activ=activ, sn=sn),
                Conv2d(input_size, input_size, kernel_size=kernel_size, stride=stride, conv=conv, pad=pad, norm=norm, activ=activ, sn=sn)
                )
    def forward(self, x):
        return x + self.block(x)
    
class VGG(nn.Module):
    def __init__(self, pool='max'):
        super(VGG, self).__init__()
        #vgg modules
        self.conv1_1 = nn.Conv2d(3, 64, kernel_size=3, padding=1)
        self.conv1_2 = nn.Conv2d(64, 64, kernel_size=3, padding=1)
        self.conv2_1 = nn.Conv2d(64, 128, kernel_size=3, padding=1)
        self.conv2_2 = nn.Conv2d(128, 128, kernel_size=3, padding=1)
        self.conv3_1 = nn.Conv2d(128, 256, kernel_size=3, padding=1)
        self.conv3_2 = nn.Conv2d(256, 256, kernel_size=3, padding=1)
        self.conv3_3 = nn.Conv2d(256, 256, kernel_size=3, padding=1)
        self.conv4_1 = nn.Conv2d(256, 512, kernel_size=3, padding=1)
        self.conv4_2 = nn.Conv2d(512, 512, kernel_size=3, padding=1)
        self.conv4_3 = nn.Conv2d(512, 512, kernel_size=3, padding=1)
        self.conv5_1 = nn.Conv2d(512, 512, kernel_size=3, padding=1)
        self.conv5_2 = nn.Conv2d(512, 512, kernel_size=3, padding=1)
        self.conv5_3 = nn.Conv2d(512, 512, kernel_size=3, padding=1)
        self.fc6 = nn.Linear(25088, 4096, bias=True)
        self.fc7 = nn.Linear(4096, 4096, bias=True)
        self.fc8_101 = nn.Linear(4096, 101, bias=True)
        if pool == 'max':
            self.pool1 = nn.MaxPool2d(kernel_size=2, stride=2)
            self.pool2 = nn.MaxPool2d(kernel_size=2, stride=2)
            self.pool3 = nn.MaxPool2d(kernel_size=2, stride=2)
            self.pool4 = nn.MaxPool2d(kernel_size=2, stride=2)
            self.pool5 = nn.MaxPool2d(kernel_size=2, stride=2)
        elif pool == 'avg':
            self.pool1 = nn.AvgPool2d(kernel_size=2, stride=2)
            self.pool2 = nn.AvgPool2d(kernel_size=2, stride=2)
            self.pool3 = nn.AvgPool2d(kernel_size=2, stride=2)
            self.pool4 = nn.AvgPool2d(kernel_size=2, stride=2)
            self.pool5 = nn.AvgPool2d(kernel_size=2, stride=2)
            
    def forward(self, x):
        out = {}
        out['r11'] = F.relu(self.conv1_1(x))
        out['r12'] = F.relu(self.conv1_2(out['r11']))
        out['p1'] = self.pool1(out['r12'])
        out['r21'] = F.relu(self.conv2_1(out['p1']))
        out['r22'] = F.relu(self.conv2_2(out['r21']))
        out['p2'] = self.pool2(out['r22'])
        out['r31'] = F.relu(self.conv3_1(out['p2']))
        out['r32'] = F.relu(self.conv3_2(out['r31']))
        out['r33'] = F.relu(self.conv3_3(out['r32']))
        out['p3'] = self.pool3(out['r33'])
        out['r41'] = F.relu(self.conv4_1(out['p3']))
        out['r42'] = F.relu(self.conv4_2(out['r41']))
        out['r43'] = F.relu(self.conv4_3(out['r42']))
        out['p4'] = self.pool4(out['r43'])
        out['r51'] = F.relu(self.conv5_1(out['p4']))
        out['r52'] = F.relu(self.conv5_2(out['r51']))
        out['r53'] = F.relu(self.conv5_3(out['r52']))
        out['p5'] = self.pool5(out['r53'])
        out['p5'] = out['p5'].view(out['p5'].size(0),-1)
        out['fc6'] = F.relu(self.fc6(out['p5']))
        out['fc7'] = F.relu(self.fc7(out['fc6']))
        out['fc8'] = self.fc8_101(out['fc7'])
        return out
    
class Mod_Net(nn.Module):
    def __init__(self):
        super(Mod_Net, self).__init__()
        self.fc_mix = nn.Linear(101, 128, bias=False)
    def forward(self, x):
        b_s = x.size(0)
        z = torch.zeros(b_s,101).type_as(x).float()
        for i in range(b_s):
            z[i, x[i]]=1
        y = self.fc_mix(z)
        y = F.sigmoid(y)
        return y
    
class generator(nn.Module):
    def __init__(self, input_size = 3, output_size = 3, activ='leakyrelu'):
        super(generator, self).__init__()
        # downsampling
        self.mod = Mod_Net()
        self.down_conv1 = Conv2d(input_size, 32, kernel_size=9, stride=1, activ=activ, sn=True)
        self.down_conv2 = Conv2d(32, 64, kernel_size=3, stride=2, activ=activ, sn=True)
        self.down_conv3 = Conv2d(64, 128, kernel_size=3, stride=2, activ=activ, sn=True)
        self.down_conv4 = Conv2d(128, 128, kernel_size=3, stride=1, activ=activ, sn=True)
        self.down_resblock = nn.Sequential(
            ResBlock(128, kernel_size=3, stride=1, activ=activ, sn=True),
            ResBlock(128, kernel_size=3, stride=1, activ=activ, sn=True),
            ResBlock(128, kernel_size=3, stride=1, activ=activ, sn=True),
            ResBlock(128, kernel_size=3, stride=1, activ=activ, sn=True)
        )
        # upsampling
        self.up_conv1 = nn.Sequential(
            nn.Upsample(scale_factor=2),
            Conv2d(256, 64, kernel_size=3, stride=1, activ=activ, sn=True)
        )
        self.up_conv2 = nn.Sequential(
            nn.Upsample(scale_factor=2),
            Conv2d(128, 32, kernel_size=3, stride=1, activ=activ, sn=True)
        )
        self.up_conv3 = nn.Sequential(
            nn.ReflectionPad2d(4),
            nn.Conv2d(32, output_size, kernel_size=9, stride=1)
        )

    def forward(self, image, age, age_modif):
        # x : (batch size, input_channel, image_width, image_height)
        # modulation network
        age = self.mod(age)
        age_modif = self.mod(age_modif)
        bs, channel = age.size()
        age_vec = age.view(bs, channel, 1, 1)
        age_modif_vec = age_modif.view(bs, channel, 1, 1)
        # encoding
        out = self.down_conv1(image) # (batch size, 32, image_w, image_h)
        skip_layer1 = self.down_conv2(out) # (batch size, 64, image_w, image_h)
        skip_layer2 = self.down_conv3(skip_layer1) # (batch size, 128, image_w, image_h)
        out = self.down_conv4(skip_layer2) # (batch size, 256, image_w, image_h)
        out = self.down_resblock(out) # (batch size, 256, image_w, image_h)

        # decoding
        image_out = age_vec * out
        modif_image_out = age_modif_vec * out

        image_out = torch.cat((image_out, skip_layer2), 1)
        modif_image_out = torch.cat((modif_image_out, skip_layer2), 1)
        image_out = self.up_conv1(image_out)
        modif_image_out = self.up_conv1(modif_image_out)
        image_out = torch.cat((image_out, skip_layer1), 1)
        modif_image_out = torch.cat((modif_image_out, skip_layer1), 1)
        image_out = self.up_conv2(image_out)
        modif_image_out = self.up_conv2(modif_image_out)
        image_out = self.up_conv3(image_out)
        modif_image_out = self.up_conv3(modif_image_out)
        return image_out, modif_image_out

class Discriminator(nn.Module):
    def __init__(self, input_size=3):
        super(Discriminator, self).__init__()
        self.conv = nn.Sequential(
            Conv2d(input_size, 32, kernel_size=4, stride=2, norm='none', activ='leakyrelu', sn=True),
            Conv2d(32, 64, kernel_size=4, stride=2, norm='batch', activ='leakyrelu', sn=True),
            Conv2d(64, 128, kernel_size=4, stride=2, norm='batch', activ='leakyrelu', sn=True),
            Conv2d(128, 256, kernel_size=4, stride=2, norm='batch', activ='leakyrelu', sn=True),
            Conv2d(256, 512, kernel_size=4, stride=1, norm='batch', activ='leakyrelu', sn=True),
            Conv2d(512, 1, kernel_size=4, stride=1, norm='none', activ='none', sn=True)
        )
    def forward(self, x):
        out = self.conv(x)
        return out