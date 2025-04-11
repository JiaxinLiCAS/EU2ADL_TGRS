# -*- coding: utf-8 -*-
"""
❗❗❗❗❗❗李嘉鑫 作者微信 BatAug 欢迎加微信交流
空天信息创新研究院20-25直博生，导师高连如
"""
"""
❗❗❗❗❗❗#此py作用：没有PSOS-Net的融合网络
"""



import torch
import torch.nn
import itertools
import hues
import numpy as np
from . import network
from collections import OrderedDict
from .read_data import readdata


class Fusion_simple(readdata):
    
    def __init__(self,args,psf_est=None,srf_est=None):
        #self.args = args在父类readdata定义了
        
        super().__init__(args)
        self.hs_bands = self.srf_gt.shape[0]
        self.ms_bands = self.srf_gt.shape[1]
 
        #获取SRF and PSF
        if psf_est is not None :
            self.psf=psf_est
        else:
            self.psf=self.psf_gt
        self.psf = np.reshape(self.psf, newshape=(1, 1, self.args.scale_factor, self.args.scale_factor)) #1 1 ratio ratio 大小的tensor
        self.psf = torch.tensor(self.psf).to(self.args.device).float()
        
        if srf_est is not None:
            self.srf=srf_est
        else:
            self.srf=self.srf_gt
        self.srf = np.reshape(self.srf.T, newshape=(self.ms_bands, self.hs_bands, 1, 1)) #self.srf.T 有一个T转置
        self.srf = torch.tensor(self.srf).to(self.args.device).float()             # ms_band hs_bands 1 1 的tensor
        #获取SRF and PSF
        
        #初始化network
        self.initialize_network()
        #初始化loss
        self.initialize_loss()
        #初始化optimizer 和 schedulers
        self.initialize_optimizer_scheduler()
        #输出模型参数个数
        self.get_information()
        self.print_parameters()
        
    def get_information(self):
        #self.model_names = ['net_hr_msi_stream', 'net_lr_hsi_stream', 'net_shared_stream', 'net_abun2hrmsi', 'net_abun2lrhsi']
        self.model_names = ['net_hr_msi_stream', 'net_lr_hsi_stream', 'net_abun2hrmsi', 'net_abun2lrhsi']
        
        self.loss_names = ['loss_hr_msi_rec'             ,  'loss_lr_hsi_rec', 
                           'loss_hr_msi_from_hrhsi'       , 'loss_lr_hsi_from_hrhsi',
                           'loss_abundance_sum2one_hrmsi' , 'loss_abundance_sum2one_lrhsi',
                           'loss_abundance_rec'
                           ]
        

        self.visual_names = ['tensor_lr_hsi','lr_hsi_rec', 
                             'tensor_hr_msi','hr_msi_rec',
                             'tensor_gt','gt_est']
        
        self.visual_corresponding_name={}
        self.visual_corresponding_name['tensor_lr_hsi'] = 'lr_hsi_rec'
        self.visual_corresponding_name['tensor_hr_msi'] = 'hr_msi_rec'
        self.visual_corresponding_name['tensor_gt']     = 'gt_est'
        
    def initialize_network(self):
        #初始化network
        self.net_hr_msi_stream = network.define_hr_msi_stream(input_channel=self.ms_bands,device=self.args.device,block_num=self.args.block_num,
                                                              output_channel=60,endmember_num=self.args.endmember_num,
                                                              activation=self.args.two_stream_activation)
        
        self.net_lr_hsi_stream = network.define_lr_hsi_stream(input_channel=self.hs_bands,device=self.args.device,block_num=self.args.block_num,
                                                              output_channel=60,endmember_num=self.args.endmember_num,
                                                              activation=self.args.two_stream_activation)
        
        '''
        self.net_shared_stream = network.define_shared_stream(device=self.args.device,endmember_num=self.args.endmember_num,
                                                          activation=self.args.shared_stream_activation )
        '''
        self.net_abun2hrmsi = network.define_abundance2image(output_channel=self.ms_bands,device=self.args.device,
                                                         endmember_num=self.args.endmember_num,activation=self.args.abun2img_activation)
       
      
        self.net_abun2lrhsi = network.define_abundance2image(output_channel=self.hs_bands,device=self.args.device,
                                                         endmember_num=self.args.endmember_num,activation=self.args.abun2img_activation)
        
        self.psf_down=network.PSF_down() #__call__(self, input_tensor, psf, ratio):
        self.srf_down=network.SRF_down() #__call__(self, input_tensor, srf):
        self.cliper_zeroone = network.ZeroOneClipper()    
        
    def initialize_loss(self):
        if self.args.Pixelwise_avg_crite == "No":
            self.criterionL1Loss = torch.nn.L1Loss(reduction='sum').to(self.args.device)  #reduction=mean sum
        else:
            self.criterionL1Loss = torch.nn.L1Loss(reduction='mean').to(self.args.device)
            
        self.criterionPixelwise = self.criterionL1Loss
        self.criterionSumToOne = network.SumToOneLoss().to(self.args.device)
        
    def initialize_optimizer_scheduler(self):
        #optimizer
        lr=self.args.S2_lr
        self.optimizers = []
        
        self.optimizer_hr_msi_stream = torch.optim.Adam(itertools.chain(self.net_hr_msi_stream.parameters()),
                                            lr=lr,betas=(0.9, 0.999))
        self.optimizers.append(self.optimizer_hr_msi_stream)
        
        self.optimizer_lr_hsi_stream = torch.optim.Adam(itertools.chain(self.net_lr_hsi_stream.parameters()),
                                            lr=lr,betas=(0.9, 0.999))
        self.optimizers.append(self.optimizer_lr_hsi_stream)
        '''
        self.optimizer_shared_stream = torch.optim.Adam(itertools.chain(self.net_shared_stream.parameters()),
                                            lr=lr,betas=(0.9, 0.999))
        self.optimizers.append(self.optimizer_shared_stream)
        '''
        
        self.optimizer_abun2hrmsi = torch.optim.Adam(itertools.chain(self.net_abun2hrmsi.parameters()),
                                            lr=lr,betas=(0.9, 0.999))
        self.optimizers.append(self.optimizer_abun2hrmsi)
        
        self.optimizer_abun2lrhsi = torch.optim.Adam(itertools.chain(self.net_abun2lrhsi.parameters()),
                                            lr=lr,betas=(0.9, 0.999))
        self.optimizers.append(self.optimizer_abun2lrhsi)
        
        #scheduler
        self.schedulers = [network.get_scheduler(optimizer, self.args) for optimizer in self.optimizers]
        
        
    
        
    def optimize_joint_parameters(self):
        
        #前向传播
        self.forward()
        
        #梯度清零
        for optimizer in self.optimizers:
            optimizer.zero_grad()
        
        #反向传播，求梯度
        self.backward_g_joint()
        
        for optimizer in self.optimizers:
            optimizer.step()
            
        #对端元裁剪到[0,1]
        self.net_abun2hrmsi.apply(self.cliper_zeroone)
        self.net_abun2lrhsi.apply(self.cliper_zeroone)
        
        
        
    def forward(self):
        
        ''' hr_msi stream  '''
        self.hr_msi_abundance=self.net_hr_msi_stream(self.tensor_hr_msi)
        
        ''' lr_hsi stream  '''
        self.lr_hsi_abundance=self.net_lr_hsi_stream(self.tensor_lr_hsi)
        
        ''' share stream  '''
        #self.hr_msi_share_abundance = self.net_shared_stream(self.hr_msi_specific_abundance)
        #self.lr_hsi_share_abundance = self.net_shared_stream(self.lr_hsi_specific_abundance)
        
        ''' plus to get hr_ms abundance '''
        #self.hr_msi_abundance = self.hr_msi_specific_abundance + self.hr_msi_share_abundance
        
        ''' plus to get lr_hsi abundance '''
        #self.lr_hsi_abundance = self.lr_hsi_specific_abundance + self.lr_hsi_share_abundance
        
        '''
        if self.args.plus_activation =='sigmoid':
            
            self.hr_msi_abundance = torch.sigmoid(self.hr_msi_abundance)
            self.lr_hsi_abundance = torch.sigmoid(self.lr_hsi_abundance)
            
        if self.args.plus_activation =='clamp':
            
            self.hr_msi_abundance.clamp_(0,1)
            self.lr_hsi_abundance.clamp_(0,1)
            
        if self.args.plus_activation =='softmax':
            self.hr_msi_abundance = torch.nn.Softmax(dim=1)(self.hr_msi_abundance)
            self.lr_hsi_abundance = torch.nn.Softmax(dim=1)(self.lr_hsi_abundance)
        
        
        else:     #'No' 不需要激活函数
            pass
        '''
        
        ''' hr_msi_abundance 2 hr_msi '''
        self.hr_msi_rec = self.net_abun2hrmsi(self.hr_msi_abundance)
        
        ''' lr_hsi_abundance 2 lr_hsi '''
        self.lr_hsi_rec =  self.net_abun2lrhsi(self.lr_hsi_abundance)
        
        ''' generate hrhsi_est '''
        self.gt_est= self.net_abun2lrhsi(self.hr_msi_abundance)
        
        ''' generate hr_msi_est '''
        self.hr_msi_from_hrhsi = self.srf_down(self.gt_est,self.srf)

        ''' generate lr_hsi_est '''
        self.lr_hsi_from_hrhsi = self.psf_down(self.gt_est, self.psf, self.args.scale_factor)
        
        ''' abundance '''
        self.lr_hsi_abundance_est = self.psf_down(self.hr_msi_abundance, self.psf, self.args.scale_factor)
        
       

        

    def backward_g_joint(self):
        '''
        self.loss_names = ['loss_hr_msi_rec'             ,  'loss_lr_hsi_rec', 
                           'loss_hr_msi_from_hrhsi'       , 'loss_lr_hsi_from_hrhsi',
                           'loss_abundance_sum2one_hrmsi' , 'loss_abundance_sum2one_lrhsi',
                           'loss_abundance_rec'
                           ]
        '''
        #lambda_A hr_msi 重建误差
        self.loss_hr_msi_rec=self.criterionPixelwise(self.tensor_hr_msi,self.hr_msi_rec)
        self.loss_hr_msi_rec_ceo=self.loss_hr_msi_rec*self.args.lambda_A
        
        #lambda_B lr_hsi 重建误差
        self.loss_lr_hsi_rec=self.criterionPixelwise(self.tensor_lr_hsi,self.lr_hsi_rec)
        self.loss_lr_hsi_rec_ceo=self.loss_lr_hsi_rec*self.args.lambda_B
        
        #lambda_C hr_msi_from_hrhsi \ lr_hsi_from_hrhsi 从预测的hr_hsi恢复到hr_msi\lr_hsi的误差
        self.loss_hr_msi_from_hrhsi=self.criterionPixelwise(self.tensor_hr_msi,self.hr_msi_from_hrhsi)
        self.loss_lr_hsi_from_hrhsi=self.criterionPixelwise(self.tensor_lr_hsi,self.lr_hsi_from_hrhsi)

        self.loss_degradation_ceo=(self.loss_hr_msi_from_hrhsi + self.loss_lr_hsi_from_hrhsi)*self.args.lambda_C
        
        #lambda_D abundance_rec 两个丰度之间的误差 【这里判断使用L1损失，还是perceptual loss】
        if self.args.use_perceptual_loss == 'Yes':

            from .network import VGGPerceptualLoss
            
            self.loss_abundance_rec=VGGPerceptualLoss(self.lr_hsi_abundance_est,self.lr_hsi_abundance,self.args) #使用perceptual loss
            self.loss_abundance_rec_ceo=self.loss_abundance_rec*self.args.lambda_D
        else:
            self.loss_abundance_rec=self.criterionPixelwise(self.lr_hsi_abundance_est,self.lr_hsi_abundance)
            self.loss_abundance_rec_ceo=self.loss_abundance_rec*self.args.lambda_D
        
        #lambda_E abundance_sum2one
        self.loss_abundance_sum2one_hrmsi = self.criterionSumToOne(self.hr_msi_abundance)
        self.loss_abundance_sum2one_lrhsi = self.criterionSumToOne(self.lr_hsi_abundance)
        
        self.loss_abundance_sum2one_ceo = (self.loss_abundance_sum2one_hrmsi + self.loss_abundance_sum2one_lrhsi)*self.args.lambda_E
        

        self.loss_all = self.loss_hr_msi_rec_ceo  + self.loss_lr_hsi_rec_ceo  + self.loss_degradation_ceo + \
                          self.loss_abundance_rec_ceo + self.loss_abundance_sum2one_ceo
                             
        #self.loss_degradation.backward(retain_graph=True)
        self.loss_all.backward(retain_graph=True)

    
    def print_parameters(self):
         for name in self.model_names:  #model_name 定义在 def initialize_network(self)里面
             if isinstance(name, str):
                 net = getattr(self,name)
                 num_params = 0
                 for param in net.parameters():
                     num_params += param.numel()
                 print('[Network %s] Total number of parameters : %.0f ' % (name, num_params))
         print('-----------------------------------------------')    
         
    def update_learning_rate(self):
        lr = self.optimizers[0].param_groups[0]['lr']
        #print('learning rate = %.7f' % lr)
        #print('-----------------------------------------------')    
        for scheduler in self.schedulers:
             if self.args.lr_policy == 'plateau':
                 scheduler.step()
             else:
                 scheduler.step()
         

    def get_current_visuals(self):
        visual_ret = OrderedDict()
        for name in self.visual_names:
            if isinstance(name, str):
                visual_ret[name] = getattr(self, name)
        return visual_ret


    # return traning losses/errors. train.py will print out these errors as debugging information
    def get_current_losses(self):
        errors_ret = OrderedDict()
        for name in self.loss_names:
            if isinstance(name, str):
                # float(...) works for both scalar tensor and float number
                errors_ret[name] = float(getattr(self, name))
        return errors_ret
    
    def get_LR(self):
        lr = self.optimizers[0].param_groups[0]['lr'] 
        return lr

    

if __name__ == "__main__":
    
    pass
