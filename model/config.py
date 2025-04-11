# -*- coding: utf-8 -*-
"""
❗❗❗❗❗❗李嘉鑫 作者微信 BatAug 欢迎加微信交流
空天信息创新研究院20-25直博生，导师高连如
"""
"""
❗❗❗❗❗❗#此py作用：本方法所有的参数 包括数据读取地址 以及 超参数等
"""

import argparse
import torch
import os
import datetime

parser = argparse.ArgumentParser(formatter_class=argparse.ArgumentDefaultsHelpFormatter)

###读取数据的地址参数
'''Whisper=8 Houston18=10 Chikusei=16'''
parser.add_argument('--scale_factor',type=int,default=12, help='仿真LrHSI时选取的缩放系数')
parser.add_argument('--data_name',type=str, default="TG",help='Whisper Houston18 TG Chikusei  ')
parser.add_argument('--sp_root_path',type=str, default='data/EU2ADL/spectral_response/',help='where you store your own spectral response')
parser.add_argument('--default_datapath',type=str, default="data/EU2ADL/",help='where you store your HSI data file and spectral response file')
parser.add_argument("--gpu_ids", type=str, default='0', help='设定GPU')

#本方法涉及到的一些超参数
parser.add_argument('--epoch_count', type=int, default=1, help='the starting epoch count, we save the model by <epoch_count>, <epoch_count>+<save_latest_freq>, ...')
parser.add_argument('--niter', type=int, default=7000, help='前7K轮保持S2_lr学习率不变')
parser.add_argument('--niter_decay', type=int, default=7000, help='后7K轮S2_lr学习线性降为0')
parser.add_argument('--lr_policy', type=str, default='lambda', help='learning rate policy: lambda|step|plateau')
#下面三个lr_decay是针对lr_policy=step|plateau时候需要设置的
parser.add_argument('--lr_decay_iters', type=int, default=100, help='multiply by a gamma every lr_decay_iters iterations')
parser.add_argument('--lr_decay_gamma', type=float, default=0.8)
parser.add_argument('--lr_decay_patience', type=int, default=50)
parser.add_argument('--print_freq', type=int, default=10)#每X轮输出一次结果


#针对Visdom可视化包需要调整的参数，只需要针对不同数据和配置修改 --display_port 即可
parser.add_argument('--display_ncols', type=int, default=2, help='if positive, display all images in a single visdom web panel with certain number of images per row.')
parser.add_argument('--display_winsize', type=int, default=256, help='display window size')
parser.add_argument('--display_id', type=int, default=1, help='window id of the web display')
parser.add_argument('--display_server', type=str, default="http://localhost", help='visdom server of the web display')
parser.add_argument('--display_env', type=str, default='main', help='visdom display environment name (default is "main")')
#这个参数很重要 指定可视化的网站
parser.add_argument('--display_port', type=int, default=8500, help='visdom port of the web display')


###
parser.add_argument("--S1_lr", type=float, default=0.001) #针对退化网络的lr
parser.add_argument("--S2_lr", type=float, default=0.0011) # 针对融合重建网络的lr



parser.add_argument('--two_stream_activation', type=str, default="No", help='sigmoid,softmax,clamp,No')
parser.add_argument('--shared_stream_activation', type=str, default="No", help='sigmoid,softmax,clamp,No')
parser.add_argument('--plus_activation', type=str, default="clamp", help='sigmoid,softmax,clamp,No')
parser.add_argument('--abun2img_activation', type=str, default="clamp", help='sigmoid,clamp')

parser.add_argument('--Pixelwise_avg_crite', type=str, default="No", help='')


#不同损失的权重
parser.add_argument('--lambda_A', type=float, default=100,help='hr_msi 重建误差')   #100
parser.add_argument('--lambda_B', type=float, default=100,help='lr_hsi 重建误差')    #100
parser.add_argument('--lambda_C', type=float, default=10,help='从预测的hr_hsi恢复到hr_msi\lr_hsi的误差')  #10
parser.add_argument('--lambda_D', type=float, default=1,help='两个丰度之间的误差')  #1
parser.add_argument('--lambda_E', type=float, default=0.01,help='丰度和为1误差')  #0.01

#参数分析
parser.add_argument('--block_num', type=int, default=3, help='SSTS里面模块数量 默认是3')
parser.add_argument('--endmember_num', type=int, default=160, help='端元的个数')

#消融实验
#是否使用本文提出的感知损失，使用的时候运行速度会变慢
parser.add_argument('--use_perceptual_loss', type=str, default="No", help='Yes ,No 是否使用本文提出的感知损失，使用的时候运行速度会变慢')
#是否使用本文提出的PSOS-Net
parser.add_argument('--use_semi_siamese', type=str, default="Yes", help='Yes ,No 是否使用使用PSOS-Net')
#是否使用本文提出ADLnet实现PSF SRF推断
parser.add_argument('--blind', type=str, default="Yes", help='Yes ,No')

#添加噪声
parser.add_argument('--noise', type=str, default="No", help='Yes ,No 仿真时是否添加噪声')
parser.add_argument('--nSNR', type=int, default=25)




args=parser.parse_args()

device = torch.device(  'cuda:{}'.format(args.gpu_ids)  ) if  torch.cuda.is_available() else torch.device('cpu') 
args.device=device
# Because the full width at half maxima of Gaussian function used to generate the PSF is set to scale factor in our experiment, 
# there exists the following relationship between  the standard deviation and scale_factor :
args.sigma = args.scale_factor / 2.35482

t=str(datetime.datetime.now().day)+str("_") + str(datetime.datetime.now().hour)+str("_") + str(datetime.datetime.now().minute)+str("_") + \
                   str(datetime.datetime.now().second)

'''针对每个数据和不同配置，所有结果存储到以下名称的文件夹中 '''
args.expr_dir=os.path.join('checkpoints', args.data_name+'_SF'+str(args.scale_factor)+'_endnum'+str(args.endmember_num)+\
                           '_A'+str(args.lambda_A)+'_B'+str(args.lambda_B)+'_C'+str(args.lambda_C)+\
                           '_D'+str(args.lambda_D)+'_E'+str(args.lambda_E) + '_' + 'block_num_'+str(args.block_num) +\
                            '_use_per' + args.use_perceptual_loss + '_use_semi' + args.use_semi_siamese + '_blind' + args.blind
                           )


