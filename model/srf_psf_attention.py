# -*- coding: utf-8 -*-
"""
❗❗❗❗❗❗李嘉鑫 作者微信 BatAug 欢迎加微信交流
空天信息创新研究院20-25直博生，导师高连如
"""
"""
❗❗❗❗❗❗#此py作用：估计PSF和SRF
"""

import numpy as np
import scipy.io as sio
import os
import torch
import torch.nn.functional as fun
import torch.nn as nn
import torch.optim as optim
from .read_data import readdata

import random


def compute_sam(x_true, x_pred):
    assert x_true.ndim ==3 and x_true.shape == x_pred.shape

    w, h, c = x_true.shape
    x_true = x_true.reshape(-1, c)
    x_pred = x_pred.reshape(-1, c)

    #sam = (x_true * x_pred).sum(axis=1) / (np.linalg.norm(x_true, 2, 1) * np.linalg.norm(x_pred, 2, 1)+1e-5) 原本的
    #sam = (x_true * x_pred).sum(axis=1) / (np.linalg.norm(x_true, 2, 1) * np.linalg.norm(x_pred, 2, 1))
    sam = (x_true * x_pred).sum(axis=1) / (np.linalg.norm(x_true, 2, 1) * np.linalg.norm(x_pred, 2, 1)+1e-7)
    sam = np.arccos(sam) * 180 / np.pi
    mSAM = sam.mean()
    #SAM_map = sam.reshape(w,h)
    
    return mSAM

def compute_psnr(x_true, x_pred):
    assert x_true.ndim == 3 and x_pred.ndim ==3

    img_w, img_h, img_c = x_true.shape
    ref = x_true.reshape(-1, img_c)
    #print(ref)
    tar = x_pred.reshape(-1, img_c)
    msr = np.mean((ref - tar)**2, 0) #列和
    #print(msr)
    max2 = np.max(ref,0)**2
    #print(max2)
    psnrall = 10*np.log10(max2/msr)
    #print(psnrall)
    m_psnr = np.mean(psnrall)
    #print(m_psnr)
    psnr_all = psnrall.reshape(img_c)
    #print( psnr_all)
    return m_psnr


def compute_ergas(x_true, x_pred, scale_factor):
    assert x_true.ndim == 3 and x_pred.ndim ==3 and x_true.shape == x_pred.shape
    
    img_w, img_h, img_c = x_true.shape

    err = x_true - x_pred
    ERGAS = 0
    for i in range(img_c):
        ERGAS = ERGAS + np.mean(  err[:,:,i] **2 / np.mean(x_true[:,:,i]) ** 2  )
    
    ERGAS = (100 / scale_factor) * np.sqrt((1/img_c) * ERGAS)
    return ERGAS


def compute_cc(x_true, x_pred):
    img_w, img_h, img_c = x_true.shape
    result=np.ones((img_c,))
    for i in range(0,img_c):
        CCi=np.corrcoef(x_true[:,:,i].flatten(),x_pred[:,:,i].flatten())
        result[i]=CCi[0,1]
        #print('result[i]',result[i])
        #print('CCi',CCi)
    #print(result)
    return result.mean()

def compute_rmse(x_true, x_pre):
     img_w, img_h, img_c = x_true.shape
     return np.sqrt(  ((x_true-x_pre)**2).sum()/(img_w*img_h*img_c)   )
 
    

class Predict_lr_msi_fhsi(nn.Module):
    def __init__(self, hs_bands,ms_bands):
        super(Predict_lr_msi_fhsi, self).__init__()
        self.hs_bands=hs_bands
        self.ms_bands=ms_bands
        
        r = 2
        self.layer1 = nn.ModuleList()
        
        
        for i in range(self.ms_bands):
            self.layer1.append(
                
               nn.Sequential(
                   nn.AdaptiveAvgPool2d(1),
                   nn.Conv2d(in_channels=hs_bands, out_channels=int(hs_bands/r), kernel_size=1)  ,
                   #nn.Sigmoid(),
                   nn.ReLU(inplace=True),
                   #nn.Conv2d(1,1,kernel_size=3,stride=1,padding=1,bias=False),
                   #nn.Sigmoid(),
                   nn.Conv2d(in_channels=int(hs_bands/r), out_channels=hs_bands, kernel_size=1),
                   nn.Softmax(dim=1)
                             )
                              ) 
        

    def forward(self, input):
            SRF_all=[] #ms_band个 1 X 46 X 1 X 1大小的SRF
            out=[]
            for i in range(self.ms_bands):
                SRF=self.layer1[i](input) #expand_as之前大小为1 46 1 1
                SRF_all.append(SRF) #ms_band个 1 X 46 X 1 X 1大小的SRF
                SRF_expand=SRF.expand_as(input) #expand_as之前大小为1 46 1 1 ，as之后为  1 46 30 30
                band    = torch.sum(input*SRF_expand, dim=1,keepdim=True) # 1 1 30 30
                out.append(band)
                
            output=torch.cat(out, dim=1) # 1 X ms_band X H X W
            SRF_out=torch.cat(SRF_all, dim=0) # ms_band X hs_band X 1 X 1
            return output,SRF_out
        

class Predict_lr_msi_fmsi(nn.Module):
    def __init__(self ,ms_bands,ker_size):
        super().__init__()
        #input (1,2,6,6)
        self.ms_bands=ms_bands
        #B,C,H,W=input.shape[0],input.shape[1],input.shape[2],input.shape[3]
        self.ker_size=ker_size
        self.layer1=nn.Sequential(
            nn.AdaptiveAvgPool2d(output_size=(self.ker_size, self.ker_size)), #output_size大小为PSF大小 1 X ms_band X h x w
            nn.Conv2d(self.ms_bands,1,kernel_size=1,stride=1)  , # 1 X 1 X H X W
            
            
            nn.ReLU(inplace=True)
           
        )
      
     
    
    def forward(self,input):
       attention_psf=self.layer1(input) #(1,1,6,6)    
       psf_div = torch.sum(attention_psf)       
       psf_div = torch.div(1.0, psf_div)
       
       attention_psf=attention_psf*psf_div
       psf = attention_psf.repeat(self.ms_bands, 1, 1, 1) #8X1X6X6
       
       output = fun.conv2d(input, psf, None, (self.ker_size, self.ker_size),  groups=self.ms_bands) #ratio为步长 None代表bias为0，padding默认为无
       #print("attention_psf",attention_psf.shape)
       return output,attention_psf  #(1,1,6,6)   

   
class BlindNet(nn.Module):
    def __init__(self, hs_bands, ms_bands, ker_size, ratio):
        super().__init__()
        self.hs_bands = hs_bands
        self.ms_bands = ms_bands
        self.ker_size = ker_size #8
        self.ratio = ratio #8
        
      
        
        self.attention_srf=Predict_lr_msi_fhsi(self.hs_bands,self.ms_bands)
       
        self.attention_psf=Predict_lr_msi_fmsi(self.ms_bands,self.ker_size)

    def forward(self, lr_hsi, hr_msi):
       
        self.lr_msi_fhsi,self.srf_est=self.attention_srf(lr_hsi)
        self.lr_msi_fmsi,self.psf_est=self.attention_psf(hr_msi)
        
        
        return  self.lr_msi_fhsi, self.lr_msi_fmsi


class Blind(readdata):
    def __init__(self, args):
        super().__init__(args)
       
        #self.args=args 在父类定义了
        
        self.S1_lr = self.args.S1_lr
        self.ker_size = self.args.scale_factor  #8
        self.ratio    = self.args.scale_factor 
        self.hs_bands = self.srf_gt.shape[0]
        self.ms_bands = self.srf_gt.shape[1]
      
        self.model = BlindNet(self.hs_bands, self.ms_bands, self.ker_size, self.ratio).to(self.args.device)
        self.optimizer = optim.Adam(self.model.parameters(), lr=self.S1_lr)
        

    def train(self, max_iter=4000, verb=True):
        
  
        lr_hsi, hr_msi = self.tensor_lr_hsi.to(self.args.device), self.tensor_hr_msi.to(self.args.device)
        for epoch in range(1, max_iter+1):
            lr_msi_fhsi_est, lr_msi_fmsi_est = self.model(lr_hsi, hr_msi)
           
            
            loss = torch.sum(torch.abs(lr_msi_fhsi_est - lr_msi_fmsi_est))
            
                    
            self.optimizer.zero_grad()
            loss.backward()
          
            self.optimizer.step()
           
            
            with torch.no_grad():
                if verb is True:
                    if (epoch ) % 100 == 0:
                        
                        info='epoch: %s, lr: %s, loss: %s' % (epoch, self.S1_lr, loss)
                        print(info)
                        print(self.args,info)
                        
                        lr_msi_fhsi_est_numpy=lr_msi_fhsi_est.data.cpu().detach().numpy()[0].transpose(1,2,0)
                        lr_msi_fmsi_est_numpy=lr_msi_fmsi_est.data.cpu().detach().numpy()[0].transpose(1,2,0)
                        self.lr_msi_fhsi_est_numpy=lr_msi_fhsi_est_numpy
                        self.lr_msi_fmsi_est_numpy=lr_msi_fmsi_est_numpy
                        train_message='生成的两个图像 train epoch:{} lr:{}\ntest:L1loss:{}, sam_loss:{}, psnr:{}, ergas:{}, CC:{}, rmse:{}\n'.\
                                  format(epoch,self.S1_lr,
                                         np.mean( np.abs( lr_msi_fhsi_est_numpy- lr_msi_fmsi_est_numpy ) ) ,
                                         compute_sam(lr_msi_fhsi_est_numpy, lr_msi_fmsi_est_numpy) ,
                                         compute_psnr(lr_msi_fhsi_est_numpy, lr_msi_fmsi_est_numpy) ,
                                         compute_ergas(lr_msi_fhsi_est_numpy, lr_msi_fmsi_est_numpy, self.ratio),
                                         compute_cc(lr_msi_fhsi_est_numpy, lr_msi_fmsi_est_numpy),
                                         compute_rmse(lr_msi_fhsi_est_numpy, lr_msi_fmsi_est_numpy)
                                             )
                        print(train_message)
                        
                        print('************')
                        test_message_SRF='SRF lr_msi_fhsi_est与lr_msi_fhsi epoch:{} lr:{}\ntest:L1loss:{}, sam_loss:{}, psnr:{}, ergas:{}, CC:{}, rmse:{}\n'.\
                                  format(epoch,self.S1_lr,
                                         np.mean( np.abs( self.lr_msi_fhsi- lr_msi_fhsi_est_numpy ) ) ,
                                         compute_sam(self.lr_msi_fhsi, lr_msi_fhsi_est_numpy) ,
                                         compute_psnr(self.lr_msi_fhsi, lr_msi_fhsi_est_numpy) ,
                                         compute_ergas(self.lr_msi_fhsi, lr_msi_fhsi_est_numpy,self.ratio),
                                         compute_cc(self.lr_msi_fhsi, lr_msi_fhsi_est_numpy),
                                         compute_rmse(self.lr_msi_fhsi, lr_msi_fhsi_est_numpy)
                                         )
                        print(test_message_SRF)
                        
                        print('************')
                        test_message_PSF='PSF lr_msi_fmsi_est与lr_msi_fmsi  epoch:{} lr:{}\ntest:L1loss:{}, sam_loss:{}, psnr:{}, ergas:{}, CC:{}, rmse:{}\n'.\
                                  format(epoch,self.S1_lr,
                                         np.mean( np.abs( self.lr_msi_fmsi- lr_msi_fmsi_est_numpy ) ) ,
                                         compute_sam(self.lr_msi_fmsi, lr_msi_fmsi_est_numpy) ,
                                         compute_psnr(self.lr_msi_fmsi, lr_msi_fmsi_est_numpy) ,
                                         compute_ergas(self.lr_msi_fmsi, lr_msi_fmsi_est_numpy,self.ratio),
                                         compute_cc(self.lr_msi_fmsi, lr_msi_fmsi_est_numpy),
                                         compute_rmse(self.lr_msi_fmsi, lr_msi_fmsi_est_numpy) 
                                         )
                        print(test_message_PSF)
                        
                        
                        
                        psf_info="estimated psf \n {} \n psf_gt \n{}".format(
                            np.squeeze(self.model.psf_est.data.cpu().detach().numpy()),
                            self.psf_gt 
                            )
                        #print(psf_info) 估计到的PSF
                       
                       
                        srf_info="estimated srf \n {} \n srf_gt \n{}".format(
                            np.squeeze(self.model.srf_est.data.cpu().detach().numpy()).T,
                            self.srf_gt 
                            )
                        #print(srf_info) 估计到的SRF
                        
                        
                        #####从目标高光谱空间下采样####
                        psf = self.model.psf_est.repeat(self.hs_bands, 1, 1, 1)
                        lr_hsi_est = fun.conv2d(self.tensor_gt.to(self.args.device), 
                                                psf, None, (self.ker_size, self.ker_size),  
                                                groups=self.hs_bands)

                        lr_hsi_est_numpy=lr_hsi_est.data.cpu().detach().numpy()[0].transpose(1,2,0)
                
                        print('************')
                        from_hrhsi_PSF='PSF lr_hsi_est与lr_hsi  epoch:{} lr:{}\ntest:L1loss:{}, sam_loss:{}, psnr:{}, ergas:{}, CC:{}, rmse:{}\n'.\
                                  format(epoch,self.S1_lr,
                                         np.mean( np.abs( self.lr_hsi- lr_hsi_est_numpy ) ) ,
                                         compute_sam(self.lr_hsi, lr_hsi_est_numpy) ,
                                         compute_psnr(self.lr_hsi, lr_hsi_est_numpy) ,
                                         compute_ergas(self.lr_hsi, lr_hsi_est_numpy,self.ratio),
                                         compute_cc(self.lr_hsi, lr_hsi_est_numpy),
                                         compute_rmse(self.lr_hsi, lr_hsi_est_numpy) 
                                         )
                        print(from_hrhsi_PSF) 
                        #####从目标高光谱空间下采样####
                        
                        #####从目标高光谱光谱下采样####
                        srf_est=np.squeeze(self.model.srf_est.data.cpu().detach().numpy()).T
                        w,h,c = self.gt.shape
                        if srf_est.shape[0] == c:
                            hr_msi_est_numpy = np.dot(self.gt.reshape(w*h,c), srf_est).reshape(w,h,srf_est.shape[1])
                        print('************')
                        from_hrhsi_SRF='SRF hr_msi_est与hr_msi  epoch:{} lr:{}\ntest:L1loss:{}, sam_loss:{}, psnr:{}, ergas:{}, CC:{}, rmse:{}\n'.\
                                  format(epoch,self.S1_lr,
                                         np.mean( np.abs( self.hr_msi- hr_msi_est_numpy ) ) ,
                                         compute_sam(self.hr_msi, hr_msi_est_numpy) ,
                                         compute_psnr(self.hr_msi, hr_msi_est_numpy) ,
                                         compute_ergas(self.hr_msi, hr_msi_est_numpy,self.ratio),
                                         compute_cc(self.hr_msi, hr_msi_est_numpy),
                                         compute_rmse(self.hr_msi, hr_msi_est_numpy) 
                                         )
                        print(from_hrhsi_SRF)        
                        #####从目标高光谱光谱下采样####
                        
                        
                        
                        
                        print("____________________________________________")
                       
                        
                       
        
        PATH=os.path.join(self.args.expr_dir,self.model.__class__.__name__+'.pth')
        torch.save(self.model.state_dict(),PATH)
        #self.psf = torch.tensor(torchkits.to_numpy(self.model.psf.data))
        #self.srf = torch.tensor(torchkits.to_numpy(self.model.srf.data))

    def get_save_result(self, is_save=True):
        
        psf = self.model.psf_est.data.cpu().detach().numpy() ## 1 1 15 15
        srf = self.model.srf_est.data.cpu().detach().numpy() # 8 46 1 1
        psf = np.squeeze(psf)  #15 15
        srf = np.squeeze(srf).T  # b x B 8 X 46 变为 46X8 和srf_gt保持一致
        self.psf, self.srf = psf, srf
       
        if is_save is True:
            sio.savemat(os.path.join(self.args.expr_dir , 'estimated_psf_srf.mat'), {'psf_est': psf, 'srf_est': srf})
            
    

      
if __name__ == "__main__":
    pass
