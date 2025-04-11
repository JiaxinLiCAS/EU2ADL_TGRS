"""
❗❗❗❗❗❗李嘉鑫 作者微信 BatAug 欢迎加微信交流
空天信息创新研究院20-25直博生，导师高连如
"""
"""
❗❗❗❗❗❗#此py作用：用于可视化
"""


import numpy as np
import os
import time
from . import util
from skimage.transform import resize
import collections
import pickle
from .evaluation import MetricsCal

# save image to the disk

def get_random_point(img, scale_factor):
    img_c, img_h, img_w = img.shape
    """two random point position in low resolution image """
    low_point1_h = np.random.randint(0,img_h)
    low_point1_w = np.random.randint(0,img_w)
    # low_point2_h = np.random.randint(0,img_h)
    # low_point2_w = np.random.randint(0,img_w)
    """corresponding position in high resolution image"""
    high_point1_h = low_point1_h*scale_factor
    high_point1_w = low_point1_w*scale_factor
 
    return {'1':[low_point1_h, low_point1_w]},{'1':[high_point1_h, high_point1_w]}

def convert2samesize(image_list):
    img_c = image_list[0].shape[0]
    height_max = np.array([img.shape[1] for img in image_list]).max()
    weight_max = np.array([img.shape[2] for img in image_list]).max()
    return [resize(img, (img_c, height_max, weight_max)) for img in image_list]

def get_spectral_lines(real_img, rec_img, points):
    lines = {}
    for key, value in points.items():
        lines[key] = [real_img[:,value[0],value[1]],rec_img[:,value[0],value[1]]]
    return lines

def paint_point_in_img(img, points):
    assert(len(img.shape) == 3)
    for key, value in points.items():
        img[:,value[0]-5:value[0]+5,value[1]-5:value[1]+5] = 1
    return img



def compute_sam(x_true, x_pred):
    assert x_true.ndim ==3 and x_true.shape == x_pred.shape

    c, w, h = x_true.shape
    x_true = x_true.reshape(c,-1)
    x_pred = x_pred.reshape(c,-1)

    sam = (x_true * x_pred).sum(axis=0) / (np.linalg.norm(x_true, 2, 0) * np.linalg.norm(x_pred, 2, 0))

    sam = np.arccos(sam) * 180 / np.pi
    mSAM = sam.mean()
    var_sam = np.var(sam)
    return mSAM, var_sam

def compute_psnr(img1, img2):
    assert img1.ndim == 3 and img2.ndim ==3

 
    img_c, img_w, img_h = img1.shape
    ref = img1.reshape(img_c, -1)
    tar = img2.reshape(img_c, -1)
    msr = np.mean((ref - tar)**2, 1)
    max2 = np.max(ref,1)**2
    # import ipdb
    # ipdb.set_trace()
    psnrall = 10*np.log10(max2/msr)
    out_mean = np.mean(psnrall)
    return out_mean


class Visualizer():
    def __init__(self, opt, sp_matrix):
        self.sp_matrix = sp_matrix
        self.display_id = opt.display_id
        self.win_size = opt.display_winsize
        self.name = opt.expr_dir
        self.opt = opt
        self.saved = False
        self.uni_id = 66
        if self.display_id > 0:
            import visdom
            self.ncols = opt.display_ncols
            self.vis = visdom.Visdom(server=opt.display_server, port=opt.display_port, env=opt.display_env, raise_exceptions=True)

        self.log_name = os.path.join(opt.expr_dir, 'loss_log.txt')
        self.precision_path = os.path.join(opt.expr_dir, 'precision.txt')
        self.save_psnr_sam_path = os.path.join(opt.expr_dir, "psnr_and_sam.pickle")
        #self.save_hhsi_path = os.path.join(opt.checkpoints_dir, opt.name)
        self.save_hhsi_path = opt.expr_dir
        with open(self.log_name, "a") as log_file:
            now = time.strftime("%c")
            log_file.write('================ Training Loss (%s) ================\n' % now)
        with open(self.precision_path, "a") as precision_file:
            now = time.strftime("%c")
            precision_file.write('================ Precision Log (%s) ================\n' % now)

    def reset(self):
        self.saved = False

    def throw_visdom_connection_error(self):
        print('\n\nCould not connect to Visdom server (https://github.com/facebookresearch/visdom) for displaying training progress.\nYou can suppress connection to Visdom using the option --display_id -1. To install visdom, run \n$ pip install visdom\n, and start the server by \n$ python -m visdom.server.\n\n')
        exit(1)
    
    # #将中间结果图片可视化 6张图，分别是HrMSI LrHSI HrHSI 以及对应的重建结果
    def display_current_results(self, visuals,win_id=[1]):
        if self.display_id > 0:  # show images in the browser
            ncols = self.ncols #2
            if ncols > 0:
                ncols = min(ncols, len(visuals)) #2
                h, w = next(iter(visuals.values())).shape[:2]
                table_css = """<style>
                        table {border-collapse: separate; border-spacing:4px; white-space:nowrap; text-align:center}
                        table td {width: %dpx; height: %dpx; padding: 4px; outline: 4px solid black}
                        </style>""" % (w, h)
                title = self.name
                label_html = ''
                label_html_row = ''
                images = []

                idx = 0
                for label, image in visuals.items():

                    image_numpy = util.tensor2im(image, self.sp_matrix) # 三波段图像   H X W X channel 8位图
                    label_html_row += '<td>%s</td>' % label
                    images.append(image_numpy.transpose([2, 0, 1]))  #这里又把H X W X channel图变为了 channel X H X W ，因为vis.image的要求
                    idx += 1
                    if idx % ncols == 0:
                        label_html += '<tr>%s</tr>' % label_html_row
                        label_html_row = ''
                white_image = np.ones_like(image_numpy.transpose([2, 0, 1])) * 255
                while idx % ncols != 0:
                    images.append(white_image)
                    label_html_row += '<td></td>'
                    idx += 1
                if label_html_row != '':
                    label_html += '<tr>%s</tr>' % label_html_row
                # pane col = image row

                img = images.pop()
                # img = paint_point_in_img(img, points)
                images.append(img)

                try:
                    self.vis.images(convert2samesize(images), nrow=ncols, win=self.display_id + win_id[0],
                                    padding=2, opts=dict(title=title + ' images'))
                    label_html = '<table>%s</table>' % label_html

                except ConnectionError:
                    self.throw_visdom_connection_error()

            else:
                idx = 10
                for label, image in visuals.items():
                    image_numpy = util.tensor2im(image, self.sp_matrix)
                    self.vis.image(image_numpy.transpose([2, 0, 1]), opts=dict(title=label),
                                   win=self.display_id + idx)
                    idx += 1
    
    #绘制光谱曲线 从LrHSI 选择两个点 和 HrHSI选两个点 分别可视化两条曲线
    def plot_spectral_lines(self, visuals, visual_corresponding_name=None,win_id=None):
        """get image"""
        real_hsi = visuals['tensor_gt'].data.cpu().float().numpy()[0]
        rec_hsi = visuals[visual_corresponding_name['tensor_gt']].data.cpu().float().numpy()[0]
        real_lhsi = visuals['tensor_lr_hsi'].data.cpu().float().numpy()[0]
        rec_lhsi = visuals[visual_corresponding_name['tensor_lr_hsi']].data.cpu().float().numpy()[0]
        scale_factor = real_hsi.shape[1]//real_lhsi.shape[1]
        
        """get random two points position for plot spectral lines"""
        low_points, high_points = get_random_point(real_lhsi, scale_factor)
        
        '''high resolution image spectral lines'''
        lines = get_spectral_lines(real_hsi, rec_hsi, high_points)
        len_spectral = np.arange(len(lines['1'][0]))
        self.vis.line(Y= np.column_stack([np.column_stack((line[0], line[1])) for line in lines.values()]),
                      X= np.column_stack([len_spectral] * 2*len(lines)),
                      win=self.display_id+win_id[0],
                      opts=dict(title='spectral'))

        '''low resolution image spectral lines'''
        lines = get_spectral_lines(real_lhsi, rec_lhsi, low_points)
        len_spectral = np.arange(len(lines['1'][0]))
        y_column_stack = np.column_stack([np.column_stack((line[0], line[1])) for line in lines.values()])
        self.vis.line(Y= y_column_stack,
                      X= np.column_stack([len_spectral] * (2*len(lines))),
                      win=self.display_id+win_id[1],
                      opts=dict(title='spectral_low_img'))
        
    #将PSNR 和 SAM 可视化 ，把7种指标保存到precision.txt，保存到psnr_and_sam.pickle
    def plot_psnr_sam(self, visuals, epoch,  visual_corresponding_name=None):
        image_name=self.opt.data_name
        '''psnr and sam updating with epoch'''
        real_hsi = visuals['tensor_gt'].data.cpu().float().numpy()[0]
        rec_hsi = visuals[visual_corresponding_name['tensor_gt']].data.cpu().float().numpy()[0]

        if not hasattr(self, 'plot_precision'):
            self.plot_precision = {'X':{}, 'Y':{}}
            self.win_id_dict = {}

        if image_name not in self.plot_precision['X']:
            self.plot_precision['X'][image_name] = []
            self.plot_precision['Y'][image_name] = []

        self.plot_precision['X'][image_name].append([epoch  , epoch ])
        result_sam,_ = compute_sam(real_hsi, rec_hsi)
        result_psnr = compute_psnr(real_hsi, rec_hsi)
        self.plot_precision['Y'][image_name].append([result_sam, result_psnr])
        '''save txt'''
        
        sam,psnr,ergas,cc,rmse,Ssim,Uqi=MetricsCal(real_hsi.transpose((1,2,0)),rec_hsi.transpose((1,2,0)), self.opt.scale_factor)
        write_message = "Epoch:{}  sam {},psnr {},ergas {},cc {},rmse {},Ssim {},Uqi {}".format(epoch, sam,psnr,ergas,cc,rmse,Ssim,Uqi)
        with open(self.precision_path, "a") as precision_file:
            precision_file.write('%s\n' % write_message)

        '''plot line'''
        if image_name not in self.win_id_dict:
            self.win_id_dict[image_name] = self.uni_id
            self.uni_id += 1
            print('uni_id',self.uni_id)
        try:
            self.vis.line(
                X=np.column_stack([np.row_stack(self.plot_precision['X'][image_name])]),
                Y=np.column_stack([np.row_stack(self.plot_precision['Y'][image_name])]),
                win=self.display_id+self.win_id_dict[image_name],
                opts=dict(
                    title='SAM and psnr of '+image_name,
                    legend=['SAM','PSNR']
                          ),
            )
        except ConnectionError:
            self.throw_visdom_connection_error()
        '''
        opts={
            'title': self.name + ' loss over time',
            'legend': self.plot_data['legend'],
            'xlabel': 'epoch',
            'ylabel': 'loss'},
        '''
        
        '''save'''
        if not hasattr(self, 'sava_precision'):
            self.sava_precision = collections.OrderedDict()
        if image_name not in self.sava_precision:
            self.sava_precision[image_name] = []
        self.sava_precision[image_name].append([result_sam, result_psnr])
        savefiles = open(self.save_psnr_sam_path, 'wb')
        pickle.dump(self.sava_precision, savefiles)
        savefiles.close()
        #np.save(os.path.join(self.save_hhsi_path, "real_{}.npy".format(image_name[0])), real_hsi)
        #np.save(os.path.join(self.save_hhsi_path, "rec_{}.npy".format(image_name[0])), rec_hsi)



    # #将losses可视化
    def plot_current_losses(self, epoch, losses):
        if not hasattr(self, 'plot_data'):
            self.plot_data = {'X': [], 'Y': [], 'legend': list(losses.keys())}
        self.plot_data['X'].append(epoch)
        self.plot_data['Y'].append([losses[k] for k in self.plot_data['legend']])
        try:
            self.vis.line(
                X=np.stack([np.array(self.plot_data['X'])] * len(self.plot_data['legend']), 1), #epoch行 loss个数个列 第一行为1 第二行为2
                Y=np.array(self.plot_data['Y']),                                                #epoch行 loss个数个列 第一行为第一次loss 第二行为第二次loss
                opts={
                    'title': self.name + ' loss over time',
                    'legend': self.plot_data['legend'],
                    'xlabel': 'epoch',
                    'ylabel': 'loss'},
                win=self.display_id)
        except ConnectionError:
            self.throw_visdom_connection_error()

    # #将loss保存到loss_log.txt里面 并 在控制台里输出print
    def print_current_losses(self, epoch, losses): #print_current_losses(epoch,losses) 
        message = '(epoch: %d) ' % (epoch)
        for k, v in losses.items():
            message += '%s: %.7f ' % (k, v)

        print(message)                       #输出loss
        with open(self.log_name, "a") as log_file:
            log_file.write('%s\n' % message)

    def plot_lr(self, lr, epoch):
        if not hasattr(self, 'lr'):
            self.lr = {'X': [], 'Y': []}

        self.lr['X'].append(epoch)
        self.lr['Y'].append(lr)
        try:
            self.vis.line(
                X=np.array(self.lr['X']),
                Y=np.array(self.lr['Y']),
                opts={
                    'title': 'learning rate',
                    'xlabel': 'epoch',
                    'ylabel': 'lr'},
                win=78)
        except ConnectionError:
            self.throw_visdom_connection_error()
