import os
import cv2

import torch
import numpy as np
from torch.autograd import Variable
from torch import nn
from torch.utils.data import DataLoader

from torchvision.datasets import MNIST
from torchvision import transforms as tfs
from torchvision.utils import save_image

from lib.roi_dg_data_layer.roibatchLoader import roibatchLoader
from lib.roi_dg_data_layer.roidb_DG import combined_roidb
from torch.utils.data.sampler import Sampler
import torchvision.transforms as transforms
import argparse
import setproctitle


im_tfs = tfs.Compose([
    tfs.ToTensor(),
    tfs.Normalize([0.5], [0.5]) # 标准化
])

# 定义网络
class conv_autoencoder(nn.Module):
    def __init__(self):
        super(conv_autoencoder, self).__init__()
        
        self.encoder = nn.Sequential(
            nn.Conv2d(3, 32, 3, stride=3, padding=1),  # (b, 16, 10, 10)
            nn.ReLU(True),
            nn.Conv2d(32, 16, 3, stride=2, padding=1),  # (b, 16, 5, 5)
            nn.Conv2d(16, 8, 3, stride=2, padding=1),  # (b, 8, 3, 3)
            nn.ReLU(True),
            # nn.MaxPool2d(2, stride=1)  # (b, 8, 2, 2)
        )
        
        self.decoder = nn.Sequential(
            nn.ConvTranspose2d(8, 16, 3, stride=2),  # (b, 16, 5, 5)
            nn.ReLU(True),
            nn.ConvTranspose2d(16, 32, 5, stride=3, padding=1),  # (b, 8, 15, 15)
            nn.ReLU(True),
            nn.ConvTranspose2d(32, 3, 2, stride=2, padding=1),  # (b, 1, 28, 28)
            # nn.Tanh()
        )

    def forward(self, x):
        encode = self.encoder(x)
        decode = self.decoder(encode)
        return encode, decode

class sampler(Sampler):
    def __init__(self, train_size, batch_size):
        self.num_data = train_size
        self.num_per_batch = int(train_size / batch_size)
        self.batch_size = batch_size
        self.range = torch.arange(0,batch_size).view(1, batch_size).long()
        self.leftover_flag = False
        if train_size % batch_size:
            self.leftover = torch.arange(self.num_per_batch*batch_size, train_size).long()
            self.leftover_flag = True

    def __iter__(self):
        rand_num = torch.randperm(self.num_per_batch).view(-1,1) * self.batch_size
        self.rand_num = rand_num.expand(self.num_per_batch, self.batch_size) + self.range

        self.rand_num_view = self.rand_num.view(-1)

        if self.leftover_flag:
            self.rand_num_view = torch.cat((self.rand_num_view, self.leftover),0)

        return iter(self.rand_num_view)

    def __len__(self):
        return self.num_data

def parse_args():
    """
    Parse input arguments
    """
    parser = argparse.ArgumentParser(description='Train a Fast R-CNN network')
    parser.add_argument('--dataset', dest='dataset',
                        help='training dataset',
                        default='pascal_voc', type=str)
    parser.add_argument('--epochs', dest='max_epochs',
                        help='number of epochs to train',
                        default=20, type=int)
    parser.add_argument('--part', dest='part',
                        help='test_s or test_t or test_all', default="test_t",
                        type=str)

    parser.add_argument('--save_dir', dest='save_dir',
                        help='directory to save models', default="./SaveFile/model/DAD_simple_G2",
                        type=str)
    parser.add_argument('--nw', dest='num_workers',
                        help='number of worker to load data',
                        default=0, type=int)
    parser.add_argument('--cuda', dest='cuda',
                        help='whether use CUDA',
                        action='store_true')
    parser.add_argument('--mGPUs', dest='mGPUs',
                        help='whether use multiple GPUs',
                        action='store_true')
    parser.add_argument('--bs', dest='batch_size',
                        help='batch_size',
                        default=1, type=int)

    # config optimization
    parser.add_argument('--lr', dest='lr',
                        help='starting learning rate',
                        default=0.001, type=float)
    parser.add_argument('--lr_decay_step', dest='lr_decay_step',
                        help='step to do learning rate decay, unit is epoch',
                        default=5, type=int)
    # log and diaplay
    parser.add_argument('--log_flag', dest='log_flag', # add by xmj
                        help='1:batch_loss, 2:epoch_test',
                        default=0, type=int)
    parser.add_argument('--task_name', dest='task_name', # add by xmj
                        help='The name of the task!',
                        default="AE?", type=str)

    args = parser.parse_args()

    return args

if __name__ == '__main__':    #仅作为脚本运行    

    # 读取设置
    args = parse_args()

    setproctitle.setproctitle("< xmj_%s >" %args.task_name)

    if not os.path.exists(args.save_dir):
        os.makedirs(args.save_dir)
    img_dir = './SaveFile/image/encoder/' + args.task_name + '/'
    if not os.path.exists(img_dir):
        os.makedirs(img_dir)
    
    # 数据读取
    imdb_s, roidb_s, ratio_list_s, ratio_index_s = combined_roidb("cityscape_2007_train_s")
    train_size_s = len(roidb_s)   # add flipped         image_index*2
    
    sampler_batch_s = sampler(train_size_s, args.batch_size)
    dataset_s = roibatchLoader(roidb_s, ratio_list_s, ratio_index_s, args.batch_size, \
                            imdb_s.num_classes, training=True)
    dataloader_s = torch.utils.data.DataLoader(dataset_s, batch_size=args.batch_size,
                                sampler=sampler_batch_s, num_workers=args.num_workers)
    
    im_data_s = torch.FloatTensor(1)
    im_info_s = torch.FloatTensor(1)
    num_boxes_s = torch.LongTensor(1)
    gt_boxes_s = torch.FloatTensor(1)
    # ship to cuda
    if args.cuda:
        im_data_s = im_data_s.cuda()
        im_info_s = im_info_s.cuda()
        num_boxes_s = num_boxes_s.cuda()
        gt_boxes_s = gt_boxes_s.cuda()
    # make variable
    im_data_s = Variable(im_data_s)
    im_info_s = Variable(im_info_s)
    num_boxes_s = Variable(num_boxes_s)
    gt_boxes_s = Variable(gt_boxes_s)

    # AutoEncoder = conv_autoencoder()
    # 读取储存的数据
    AutoEncoder = torch.load('./SaveFile/model/encoder/AE2.pth', map_location='cpu')
    print(AutoEncoder)
    model_dict  =  AutoEncoder.state_dict() 
    print(model_dict['encoder.0.weight'].size())
    print(model_dict['decoder.0.weight'].size())
    # print(model_dict['decoder.0.weight'][7])
    model_dict['decoder.4.weight'][31] = torch.zeros(3,2,2)
    # print(model_dict['decoder.0.weight'][7])
    
    AutoEncoder.load_state_dict(model_dict)
    if args.mGPUs:
        AutoEncoder = nn.DataParallel(AutoEncoder)

    if args.cuda:
        AutoEncoder.cuda()

    criterion = nn.MSELoss(size_average=False) 

    optimizer = torch.optim.Adam(AutoEncoder.parameters(), lr=1e-3, weight_decay=1e-5)
        
    iters_per_epoch = int(train_size_s / args.batch_size)

    for epoch in range(1, args.max_epochs + 1):

            data_iter_s = iter(dataloader_s)
            for step in range(iters_per_epoch):
                data_s = next(data_iter_s)

                im_data_s.data.resize_(data_s[0].size()).copy_(data_s[0])   #change holder size
                im_info_s.data.resize_(data_s[1].size()).copy_(data_s[1])
                gt_boxes_s.data.resize_(data_s[2].size()).copy_(data_s[2])
                num_boxes_s.data.resize_(data_s[3].size()).copy_(data_s[3])

                img_data = im_data_s

                # img_en = AutoEncoder.encoder(img_data)
                # print(img_en.size())

                # img_de = AutoEncoder.decoder(img_en)
                # print(img_de.size())
                img_en, img_de = AutoEncoder(img_data)
                img_de_r = nn.functional.upsample(img_de, size=img_data.size()[2:], mode='bilinear', align_corners=False) 
                loss = criterion(img_de_r, img_data) / img_data.shape[0]

                # 反向传播
                optimizer.zero_grad()
                loss.backward()
                optimizer.step()

                if (step+1)%100==0:
                    print('step: {}, Loss: {:.4f}'.format(step+1, loss.item()))

                if ((step+1)<1000 and (step+1)%100==0) or \
                    (step+1)%1000==0 or (step+1)<10 or \
                    ((step+1)<100 and (step+1)%10==0):
                    
                    img = img_data[0].cpu().numpy().transpose((1, 2, 0))
                    cv2.imwrite(img_dir + '%s_%s_in.jpg' %(args.task_name, step+1), img)

                    img = img_de[0].cpu().detach().numpy().transpose((1, 2, 0))
                    img = np.clip(img, 0, 255)
                    cv2.imwrite(img_dir + '%s_%s_out.jpg' %(args.task_name, step+1), img)
            
            torch.save(AutoEncoder, args.save_dir + args.task_name + ',pth') 
            img = img_de[0].cpu().detach().numpy().transpose((1, 2, 0))
            img = np.clip(img, 0, 255)
            cv2.imwrite(img_dir + "%s_e%s_out.jpg" %(args.task_name, epoch), img)