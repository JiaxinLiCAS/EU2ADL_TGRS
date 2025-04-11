# -*- coding: utf-8 -*-
"""
❗❗❗❗❗❗李嘉鑫 作者微信 BatAug 欢迎加微信交流
空天信息创新研究院20-25直博生，导师高连如
"""
"""
❗❗❗❗❗❗#此py作用：网络所需要的模块和损失
"""
import numpy as np
import torch
import torch.nn.functional as fun
import torch.nn as nn
from torch.optim import lr_scheduler
import random
from torch.nn import init
from torchvision import models

def get_scheduler(optimizer, opt):
    if opt.lr_policy == 'lambda':
        def lambda_rule(epoch):
            lr_l = 1.0 - max(0, epoch + 1 + opt.epoch_count - opt.niter) / float(opt.niter_decay + 1)
            return lr_l
        scheduler = lr_scheduler.LambdaLR(optimizer, lr_lambda=lambda_rule)
    elif opt.lr_policy == 'step':
        scheduler = lr_scheduler.StepLR(optimizer, step_size=opt.lr_decay_iters, gamma=opt.lr_decay_gamma)
    elif opt.lr_policy == 'plateau':
        scheduler = lr_scheduler.ReduceLROnPlateau(optimizer, mode='max',
                                                   factor=opt.lr_decay_gamma,
                                                   patience=opt.lr_decay_patience)
    else:
        return NotImplementedError('learning rate policy [%s] is not implemented', opt.lr_policy)
    return scheduler

'''       Loss           '''
class SumToOneLoss(nn.Module):
    def __init__(self):
        super(SumToOneLoss, self).__init__()
        self.register_buffer('one', torch.tensor(1, dtype=torch.float))
        self.loss = nn.L1Loss(reduction='sum')

    def get_target_tensor(self, input):
        target_tensor = self.one
        return target_tensor.expand_as(input)

    def __call__(self, input):
        input = torch.sum(input, dim=1) #计算每个像素位置光谱向量的和 #1 x h x w 的1矩阵
        target_tensor = self.get_target_tensor(input) #1 x h x w 的1矩阵
        # print(input[0,:,:])
        loss = self.loss(input, target_tensor)
        # loss = torch.sum(torch.abs(target_tensor - input))
        return loss
'''       Loss           '''
    
'''       模型初始化公用代码           '''
def init_weights(net, init_type, gain):
    def init_func(m):
        classname = m.__class__.__name__
        if hasattr(m, 'weight') and (classname.find('Conv') != -1 or classname.find('Linear') != -1):
            #print('classname',classname)
            if init_type == 'normal':
                init.normal_(m.weight.data, 0.0, gain)
            elif init_type == 'xavier':
                init.xavier_normal_(m.weight.data, gain=gain)
            elif init_type == 'kaiming':
                init.kaiming_normal_(m.weight.data, a=0, mode='fan_in')
            elif init_type == 'orthogonal':
                init.orthogonal_(m.weight.data, gain=gain)
            elif init_type == 'mean_space':
                batchsize, channel, height, weight = list(m.weight.data.size())
                m.weight.data.fill_(1/(height*weight))
            elif init_type == 'mean_channel':
                batchsize, channel, height, weight = list(m.weight.data.size())
                m.weight.data.fill_(1/(channel))
            else:
                raise NotImplementedError('initialization method [%s] is not implemented' % init_type)
            if hasattr(m, 'bias') and m.bias is not None:
                #print('classname',classname)
                init.constant_(m.bias.data, 0.0)
        elif classname.find('BatchNorm2d') != -1:
            init.normal_(m.weight.data, 1.0, gain)
            init.constant_(m.bias.data, 0.0)

    print('initialize network with %s' % init_type)
    net.apply(init_func)

def init_net(net, init_type, init_gain, device):
    
    net.to(device)  
    init_weights(net, init_type, gain=init_gain)
    return net
'''    模型初始化公用代码           '''
##########################################################################################################

def define_for_igarss(input_channel, endmember_num,device,init_type='kaiming', init_gain=0.02):
    net = for_igarss(input_channel, endmember_num)

    return init_net(net, init_type, init_gain,device)


class for_igarss(nn.Module):
    def __init__(self, input_channel, endmember_num):
        super(for_igarss, self).__init__()
        res=(endmember_num-input_channel)//4
        self.net = nn.Sequential(
            nn.Conv2d(input_channel, input_channel + res, 1, 1, 0), #nn.Conv2d(in_channels=3,out_channels=64,kernel_size=4,stride=2,padding=1)
            nn.ReLU(),
            nn.Conv2d(input_channel + res, input_channel + 2*res, 1, 1, 0),
            nn.ReLU(),
            nn.Conv2d(input_channel + 2*res,input_channel + 3*res, 1, 1, 0),
            nn.ReLU(),
            nn.Conv2d(input_channel + 3*res, endmember_num, 1, 1, 0),

        )

    def forward(self, x):
        
        return self.net(x)
        
    

class spatial_res_block(nn.Module):
    def __init__(self,input_channel): #input_channel 60
        super().__init__()
        assert(input_channel % 3==0)
        self.three=nn.Sequential(
        nn.Conv2d(in_channels=input_channel,out_channels=int(input_channel/3),kernel_size=1,stride=1),
        nn.ReLU(),
        nn.Conv2d(in_channels=int(input_channel/3),out_channels=int(input_channel/3),kernel_size=3,stride=1,padding=1) 
                                )
        
        self.five=nn.Sequential(
        nn.Conv2d(in_channels=input_channel,out_channels=int(input_channel/3),kernel_size=1,stride=1),
        nn.ReLU(),
        nn.Conv2d(in_channels=int(input_channel/3),out_channels=int(input_channel/3),kernel_size=5,stride=1,padding=2) 
                                )
        
        self.seven=nn.Sequential(
        nn.Conv2d(in_channels=input_channel,out_channels=int(input_channel/3),kernel_size=1,stride=1),
        nn.ReLU(),
        nn.Conv2d(in_channels=int(input_channel/3),out_channels=int(input_channel/3),kernel_size=7,stride=1,padding=3) 
                                )
    
    def forward(self,input):
        identity_data = input
        output1=self.three(input)  
        #print("output1",output1.shape) #torch.Size([1, 20, 240, 240])
        
        output2=self.five(input)
        #print("output2",output2.shape) #torch.Size([1, 20, 240, 240])
        
        output3=self.seven(input) 
        #print("output3",output3.shape) #torch.Size([1, 20, 240, 240])
        
        output=torch.cat((output1,output2,output3),dim=1) # 60
        #print("output",output.shape)  #torch.Size([1, 60, 240, 240])
        
        output = torch.add(output, identity_data)
        output = nn.ReLU()(output)
        
        return output

def define_hr_msi_stream(input_channel,device,block_num=3,output_channel=60,endmember_num=100,activation='No', init_type='kaiming', init_gain=0.02): #A
    #block_num:spatial_res_blockd的个数
    #input_channel：hr_msi的波段数
    #output_channel：self.begin输出的波段数
    #endmember_num：端元个数
    net = hr_msi_stream(input_channel, block_num, output_channel, endmember_num, activation)

    return init_net(net, init_type, init_gain,device)    
 
class hr_msi_stream(nn.Module):
    def __init__(self,input_channel,block_num,output_channel,endmember_num,activation): #output_channel 60
        #block_num:spatial_res_blockd的个数
        #input_channel：hr_msi的波段数
        #output_channel：self.begin输出的波段数
        #endmember_num：端元个数
        assert(activation in ['sigmoid','softmax','clamp','No'])
        super().__init__()
        self.activation=activation
        self.block_num=block_num
        layer = []
        
        self.begin=nn.Conv2d(in_channels=input_channel,out_channels=output_channel,kernel_size=3,stride=1,padding=1)
        for i in range(self.block_num):
            layer.append(
                                spatial_res_block(output_channel)
                              ) 
        self.spatial_blocks=nn.Sequential(*layer)
        self.end=nn.Conv2d(in_channels=output_channel,out_channels=endmember_num,kernel_size=3,stride=1,padding=1)
    
    def forward(self,input):
        output1=self.begin(input)
        #print("output1~",output1.shape) torch.Size([1, 60, 240, 240])
        output2=self.spatial_blocks(output1)
        #print("output2~",output2.shape) torch.Size([1, 60, 240, 240])
        output3=self.end(output2)
        #print("output3~",output3.shape)   torch.Size([1, 100, 240, 240])
        
        if self.activation =='sigmoid':
            return torch.sigmoid(output3)
        if self.activation =='clamp':
            return output3.clamp_(0,1)
        if self.activation =='softmax':
            return nn.Softmax(dim=1)(output3)
        else:     #'No' 不需要激活函数
            return  output3
        
        return output3 #1 endnum  H  W

##############################################################################################################################

class spectral_res_block(nn.Module):
    def __init__(self,input_channel): #input_channel 60
        super().__init__()
        self.one=nn.Sequential(
        nn.Conv2d(in_channels=input_channel,out_channels=int(input_channel/3),kernel_size=1,stride=1),
        nn.ReLU(),
        nn.Conv2d(in_channels=int(input_channel/3),out_channels=input_channel,kernel_size=1,stride=1) 
                                )
        
    def forward(self,input):
        identity_data = input
        output=self.one(input) # 60
        output = torch.add(output, identity_data) # 60
        output = nn.ReLU()(output) # 60
        return output 

def define_lr_hsi_stream(input_channel,device,block_num=3,output_channel=60,endmember_num=100,activation='No', init_type='kaiming', init_gain=0.02): #A
    #block_num:spectral_res_block的个数
    #input_channel：lr_hsi的波段数
    #output_channel：self.begin输出的波段数
    #endmember_num：端元个数
    net = lr_hsi_stream(input_channel, block_num, output_channel, endmember_num, activation)

    return init_net(net, init_type, init_gain,device) 
    
class lr_hsi_stream(nn.Module):
    def __init__(self,input_channel,block_num,output_channel,endmember_num,activation='No'): #output_channel 60
        #block_num:spectral_res_blockd的个数
        #input_channel：lr_hsi的波段数
        #output_channel：self.begin输出的波段数
        #endmember_num：端元个数
        super().__init__()
        assert(activation in ['sigmoid','softmax','clamp','No'])
        self.activation=activation
        self.block_num=block_num
        layer = []
        self.begin=nn.Conv2d(in_channels=input_channel,out_channels=output_channel,kernel_size=1,stride=1) #!!!
        for i in range(self.block_num):
            layer.append(
                                spectral_res_block(output_channel)
                              ) 
        self.spectral_blocks=nn.Sequential(*layer)
        self.end=nn.Conv2d(in_channels=output_channel,out_channels=endmember_num,kernel_size=3,stride=1,padding=1)
    
    def forward(self,input):
        output1=self.begin(input)
        #print("output1~",output1.shape) torch.Size([1, 60, 240, 240])
        output2=self.spectral_blocks(output1)
        #print("output2~",output2.shape) torch.Size([1, 60, 240, 240])
        output3=self.end(output2)
        #print("output3~",output3.shape)   torch.Size([1, 100, 240, 240])
        
        if self.activation =='sigmoid':
            return torch.sigmoid(output3)
        if self.activation =='clamp':
            return output3.clamp_(0,1)
        if self.activation =='softmax':
            return nn.Softmax(dim=1)(output3)
        else:     #'No' 不需要激活函数
            return  output3
        
        return output3 #1 endnum  h w
 
class spectral_attention(nn.Module):
    pass
    
class spatial_attention(nn.Module):
    pass

############################################################################################################################
    
def define_shared_stream(device,endmember_num=100,activation='No', init_type='kaiming', init_gain=0.02): #A
    
    #endmember_num：端元个数
    net = shared_stream(endmember_num, activation)

    return init_net(net, init_type, init_gain,device) 

class shared_stream(nn.Module):
    def __init__(self,endmember_num,activation): #output_channel 60
        assert(activation in ['sigmoid','softmax','clamp','No'])
        super().__init__()
        self.endmember_num=endmember_num
        self.activation=activation
        self.layer1=nn.Sequential(
                nn.Conv2d(in_channels=self.endmember_num, out_channels=self.endmember_num,kernel_size=3,stride=1,padding=1),
                #nn.LeakyReLU(0.2, True),
                nn.ReLU(),
                nn.Conv2d(in_channels=self.endmember_num, out_channels=self.endmember_num,kernel_size=3,stride=1,padding=1),
                #nn.LeakyReLU(0.2, True),
            )
    def forward(self,input):
        output=self.layer1(input)
        
        if self.activation =='sigmoid':
            return torch.sigmoid(output)
        if self.activation =='clamp':
            return output.clamp_(0,1)
        if self.activation =='softmax':
            return nn.Softmax(dim=1)(output)
        else:     #'No' 不需要激活函数
            return  output
        
        return output #1 endnum  h w


########################################################################################################################

def define_abundance2image(output_channel,device,endmember_num=100,activation='clamp', init_type='kaiming', init_gain=0.02): #A
    
   
    #output_channel：从丰度恢复出图像的波段数
    #endmember_num：端元个数
    net = abundance2image(endmember_num, output_channel,activation)

    return init_net(net, init_type, init_gain,device) 


class abundance2image(nn.Module): #参数充当端元
    def __init__(self, endmember_num, output_channel,activation):
        assert(activation in ['sigmoid','clamp','No'])
        super(abundance2image, self).__init__()
        self.activation=activation
        self.layer = nn.Sequential(
            nn.Conv2d(in_channels=endmember_num, out_channels=output_channel, kernel_size=1, stride=1,bias=False),
        )
        
    def forward(self, input):
        output=self.layer(input)
        
        if self.activation =='sigmoid':
            return torch.sigmoid(output)
        if self.activation =='clamp':
            return output.clamp_(0,1)
        else:     #'No' 不需要激活函数
            return  output
        
        return output #1 endnum  h w  
########################################################################################################################
    
''' PSF and SRF '''    
class PSF_down():

    def __call__(self, input_tensor, psf, ratio): #PSF为#1 1 ratio ratio 大小的tensor
        _,C,_,_=input_tensor.shape[0],input_tensor.shape[1],input_tensor.shape[2],input_tensor.shape[3]
        if psf.shape[0] == 1:
            psf = psf.repeat(C, 1, 1, 1) #8X1X8X8
                                               #input_tensor: 1X8X400X400
        output_tensor = fun.conv2d(input_tensor, psf, None, (ratio, ratio),  groups=C) #ratio为步长 None代表bias为0，padding默认为无
        return output_tensor

class SRF_down():
  
    def __call__(self, input_tensor, srf): # srf 为 ms_band hs_bands 1 1 的tensor      
        output_tensor = fun.conv2d(input_tensor, srf, None)
        return output_tensor
''' PSF and SRF '''  

''' 将参数裁剪到[0,1] '''
class ZeroOneClipper(object):

    def __call__(self, module):
        # filter the variables to get the ones you want
        if hasattr(module, 'weight'):
            w = module.weight.data
            w.clamp_(0,1)
''' 将参数裁剪到[0,1] '''


''' VGG16 '''

class Vgg16(nn.Module):
    def __init__(self):
        super(Vgg16, self).__init__()
        features = models.vgg16(pretrained=True).features
        self.to_relu_1_2 = nn.Sequential() 
        self.to_relu_2_2 = nn.Sequential() 
        self.to_relu_3_3 = nn.Sequential()
        #self.to_relu_4_3 = nn.Sequential()
        self.feature_list = [4, 9, 16]
        
        for x in range(4):
            self.to_relu_1_2.add_module(str(x), features[x])
        for x in range(4,9):
            self.to_relu_2_2.add_module(str(x), features[x])
        for x in range(9,16):
            self.to_relu_3_3.add_module(str(x), features[x])

        # don't need the gradients, just want the features
        for param in self.parameters():
            param.requires_grad = False

    def forward(self, x):
        h = self.to_relu_1_2(x)
        #print("1")
        feature1 = h
        h = self.to_relu_2_2(h)
        #print("2")
        feature2 = h
        h = self.to_relu_3_3(h)
        #print("3")
        feature3 = h
     
        out = (feature1, feature2, feature3)
        return out

def VGGPerceptualLoss(lr_hsi_abundance_est, lr_hsi_abundance,args):
    B,C,W,H=lr_hsi_abundance_est.shape
    flag=C//3
    
    lr_hsi_abundance_est_RGB   = torch.cat(
                                (
                                    torch.mean(lr_hsi_abundance_est[:, 0:flag, :, :], 1).unsqueeze(1), 
                                    torch.mean(lr_hsi_abundance_est[:,flag:flag*2, :, :], 1).unsqueeze(1), 
                                    torch.mean(lr_hsi_abundance_est[:, flag*2:, :, :], 1).unsqueeze(1)
                                ),   1)
    
    lr_hsi_abundance_RGB   = torch.cat(
                                (
                                    torch.mean(lr_hsi_abundance[:, 0:flag, :, :], 1).unsqueeze(1), # 1 X 1 X H X W
                                    torch.mean(lr_hsi_abundance[:,flag:flag*2, :, :], 1).unsqueeze(1), 
                                    torch.mean(lr_hsi_abundance[:, flag*2:, :, :], 1).unsqueeze(1)
                                ),  1) #concat 之后 1 X 3 X H X W
    
    vgg16=Vgg16().to(args.device)
    features_lr_hsi_abundance_est = vgg16(lr_hsi_abundance_est_RGB)
    features_lr_hsi_abundance     = vgg16(lr_hsi_abundance_RGB)
    
    if args.Pixelwise_avg_crite == "No":
        criterionL1Loss = torch.nn.L1Loss(reduction='sum').to(args.device)  #reduction=mean sum
    else:
        criterionL1Loss = torch.nn.L1Loss(reduction='mean').to(args.device)
    
    loss = 0
    for i in range(len(features_lr_hsi_abundance)):
        loss_i = criterionL1Loss(features_lr_hsi_abundance_est[i], features_lr_hsi_abundance[i])
        loss = loss + loss_i 
    
    return loss
    

''' VGG16 '''




if __name__ == "__main__":
    pass
    