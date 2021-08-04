from numpy.fft import fftshift
from torchvision.transforms.transforms import Lambda
from train_pytorch import BATCH_SIZE, HEIGHT, WIDTH
import torch
import torch.nn as nn
import torch.optim as optim
import numpy as np
import cv2
import os, glob
from PIL import Image
import torchvision.transforms as transforms
from models import TensorHoloModel
import random
import torch.backends.cudnn as cudnn

torch.manual_seed(0)
torch.cuda.manual_seed(0)
torch.cuda.manual_seed_all(0)
np.random.seed(0)
cudnn.benchmark = False
cudnn.deterministic = True
random.seed(0)

nm = 1e-9
um = 1e-6
mm = 1e-3
cm = 1e-2

TRAIN_RGB_DIR="C:/python/HOLOGRAM/data/train/rgb/*.png"
TRAIN_DPT_DIR="C:/python/HOLOGRAM/data/train/depth2/*.bmp"
TRAIN_AMP_DIR="C:/python/HOLOGRAM/data/train/amp/"+"*.bmp"
TRAIN_PHS_DIR="C:/python/HOLOGRAM/data/train/phs/"+"*.bmp"

TEST_RGB_DIR="C:/python/HOLOGRAM/data/test/rgb/*.png"
TEST_DPT_DIR="C:/python/HOLOGRAM/data/test/depth2/*.bmp"
TEST_AMP_DIR="C:/python/HOLOGRAM/data/test/amp/*.bmp"
TEST_PHS_DIR="C:/python/HOLOGRAM/data/test/phs/*.bmp"

CKPT_DIR="C:/python/HOLOGRAM/ckpt/"
RESULTS_DIR="C:/python/HOLOGRAM/results/"

EVAL_RGB="C:/python/HOLOGRAM/data/eval/rgb.png"
EVAL_DPT="C:/python/HOLOGRAM/data/eval/dpt.bmp"

BATCH_SIZE=6

TEST_CKPT_NAME='tensorholo.pt'

def rgb_file_list(is_test):
    if is_test==True:
        typ='test'
        num1=800
        num2=1000
    else:
        typ='train'
        num1=0
        num2=800
    rgb_list=list()
    for img_idx in range(num1, num2):
        rgb_path='./data/'+typ+'/rgb/'+str(img_idx)+'.png'
        rgb_list.append(rgb_path)
    return rgb_list

def dpt_file_list(is_test):
    if is_test==True:
        typ='test'
        num1=800
        num2=1000
    else:
        typ='train'
        num1=0
        num2=800
    dpt_list=list()
    for img_idx in range(num1, num2):
        dpt_path='./data/'+typ+'/dpt/'+str(img_idx)+'.bmp'
        dpt_list.append(dpt_path)
    return dpt_list

def amp_file_list(is_test):
    if is_test==True:
        typ='test'
        num1=800
        num2=1000
    else:
        typ='train'
        num1=0
        num2=800
    amp_list=list()
    for img_idx in range(num1, num2):
        amp_path='./data/'+typ+'/amp/Amp_'+str(img_idx)+'.bmp'
        amp_list.append(amp_path)
    return amp_list

def phs_file_list(is_test):
    if is_test==True:
        typ='test'
        num1=800
        num2=1000
    else:
        typ='train'
        num1=0
        num2=800
    phs_list=list()
    for img_idx in range(num1, num2):
        phs_path='./data/'+typ+'/phs/Phase_'+str(img_idx)+'.bmp'
        phs_list.append(phs_path)
    return phs_list    

class ImageTransform():
    def __init__(self, mean, std):
        self.data_transform=transforms.Compose([transforms.ToTensor(), transforms.Normalize(mean, std)])
    
    def __call__(self, img):
        return self.data_transform(img)

class TensorHoloDataset(torch.utils.data.Dataset):
    def __init__(self, rgb_list, dpt_list, amp_list, phs_list, transform):
        self.rgb_list=rgb_list
        self.dpt_list=dpt_list
        self.amp_list=amp_list
        self.phs_list=phs_list
        self.transform=transform
        #self.transform=None

    def __len__(self):
        return len(self.rgb_list)
    
    def __getitem__(self, index):
        rgb_path=self.rgb_list[index]
        #rgb=Image.open(rgb_path)
        rgb=cv2.imread(rgb_path)
        #rgb=rgb.convert("RGB")
        rgb=cv2.cvtColor(rgb, cv2.COLOR_BGR2RGB)
        #rgb=np.asarray(rgb, dtype=np.uint8)
        #rgb=Image.open(rgb)
        '''
        rr=rgb.copy()
        rr[:,:,1]=0
        rr[:,:,2]=0
        rr=self.transform(rr)
        '''
        rgb=self.transform(rgb)

        dpt_path=self.dpt_list[index]
        #dpt=Image.open(dpt_path)
        dpt=cv2.imread(dpt_path)
        dpt=cv2.cvtColor(dpt, cv2.COLOR_BGR2GRAY)
        dpt=self.transform(dpt)


        amp_path=self.amp_list[index]
        #amp=Image.open(amp_path)
        amp=cv2.imread(amp_path)
        #amp=amp.convert("RGB")
        amp=cv2.cvtColor(amp, cv2.COLOR_BGR2RGB)
        '''
        ar=amp.copy()
        #ar=np.asarray(ar, dtype=np.uint8)
        ar[:,:,1]=0
        ar[:,:,2]=0
        ar=self.transform(ar)
        '''
        amp=self.transform(amp)

        phs_path=self.phs_list[index]
        #phs=Image.open(phs_path)
        phs=cv2.imread(phs_path)
        #phs=phs.convert("RGB")
        phs=cv2.cvtColor(phs, cv2.COLOR_BGR2RGB)
        '''
        pr=phs.copy()
        #pr=np.asarray(pr, dtype=np.uint8)
        pr[:,:,1]=0
        pr[:,:,2]=0
        pr=self.transform(pr)
        '''
        phs=self.transform(phs)
        
        #return rr, dpt, ar, pr
        return rgb, dpt, amp, phs

def rescale(cgh):
    min_cgh = np.min(cgh)
    max_cgh = np.max(cgh)

    cgh = (cgh - min_cgh) / (max_cgh - min_cgh)
    return cgh

device='cuda'

net=TensorHoloModel()
net=net.to(device)
net=torch.nn.DataParallel(net)

learning_rate=1e-4
filename='tensorholo.pt'

criterion = nn.MSELoss()
optimizer=optim.Adam(net.parameters(), lr=learning_rate, betas=(0.9, 0.999))

rgb_list=rgb_file_list(is_test=False)
dpt_list=dpt_file_list(is_test=False)
amp_list=amp_file_list(is_test=False)
phs_list=phs_file_list(is_test=False)
'''
dataset=TensorHoloDataset(rgb_list=rgb_list, dpt_list=dpt_list, amp_list=amp_list, phs_list=phs_list, transform=transforms.Compose([transforms.ToTensor()]))
loader=torch.utils.data.DataLoader(dataset, batch_size=BATCH_SIZE, shuffle=True)
meanRGB=[np.mean(x.numpy(), axis=(1,2)) for x, _ in loader]
mean=np.mean([m[0] for m in meanRGB])
print(mean)
mean=dataset.mean(axis=(0,1,2))
std=dataset.std(axis=(0,1,2))

mean=mean/255.0
std=std/255.0

print(mean)
print(std)
'''
def train(epoch):
    train_dataset=TensorHoloDataset(rgb_list=rgb_list, dpt_list=dpt_list, amp_list=amp_list, phs_list=phs_list, transform=transforms.Compose([transforms.ToTensor()]))
    #train_dataset=TensorHoloDataset(rgb_list=rgb_list, dpt_list=dpt_list, amp_list=amp_list, phs_list=phs_list, transform=None)
    train_loader=torch.utils.data.DataLoader(train_dataset, batch_size=BATCH_SIZE, shuffle=True)


    print('\n[ Train Epoch: %d ]'%epoch)
    net.train()
    train_loss=0

    for batch_idx, (train_rgb, train_dpt, train_amp, train_phs) in enumerate(train_loader):
        train_rgb=train_rgb.to(device)
        train_dpt=train_dpt.to(device)
        train_amp=train_amp.to(device)
        train_phs=train_phs.to(device)
        optimizer.zero_grad()

        output_amp, output_phs=net(train_rgb, train_dpt)
        output_amp=output_amp[None, :, :, :]
        output_phs=output_phs[None, :, :, :]
        loss_amp=criterion(output_amp, train_amp)
        loss_phs=criterion(output_phs, train_phs)
        #delta=torch.atan2(torch.sin(output_phs-train_phs), torch.cos(output_phs-train_phs))
        #criterion1=output_amp-train_amp*torch.exp(1j*(delta-delta.conj()))
        #criterion1=torch.linalg.norm(criterion1, ord=2, axis=1)
        '''
        WIDTH=192
        HEIGHT=192
        near=0.00000000000000001*cm
        far=1*mm
        pitch_w=1/(8*um)
        pitch_h=1/(8*um)
        BETA=0.35

        sm=np.arange(-WIDTH/2, WIDTH/2)
        sn=np.arange(-HEIGHT/2, HEIGHT/2)
        m,n = np.meshgrid(sm, sn)
        lambda_=[638*nm, 520*nm, 450*nm]
            
        def ASM(H_target, dt):
            kernel=np.exp(2j*np.pi*(dt)*np.sqrt(lambda_**(-2)-(m/pitch_w)**2-(n/pitch_h)**2))
            # phase_shift = 2 * (np.pi * (1 / self.wavelengths) * tf.sqrt(1. - (self.wavelengths * self.fx) ** 2 - (self.wavelengths * self.fy) ** 2))
            u1=np.fft.fftshift(np.fft.fft2(H_target))
            return np.fft.fftshift(np.fft.ifft2(np.fft.ifftshift(u1*kernel)))
        criterion2+=np.exp(BETA*(2*((far-near)/2)-(dt-Dt)))*(np.absolute(ASM(output_amp*np.exp(1j*output_phs), dt))-np.absolute(ASM(train_amp*np.exp(1j*train_phs), dt))+np.gradient(np.absolute(ASM(output_amp*np.exp(1j*output_phs) ,dt)))-np.gradient(np.absolute(ASM(train_amp*np.exp(1j*train_phs), dt))))
        criterion2=np.linalg.norm(criterion2, axis=1, ord=1)
        '''
        loss=loss_amp+loss_phs
        #loss=torch.sum(criterion1)
        loss.backward()

        optimizer.step()
        train_loss+=loss.item()

        if batch_idx %10==0:
            print('Current batch:', str(batch_idx))
            print('Current train loss:', loss.item())
    print('Total train loss:', train_loss/800) # 800 -> len(train data)

def test(epoch):
    rgb_list=rgb_file_list(is_test=True)
    dpt_list=dpt_file_list(is_test=True)
    amp_list=amp_file_list(is_test=True)
    phs_list=phs_file_list(is_test=True)
    mean=(0.4,)
    std=(0.2,)
    test_dataset=TensorHoloDataset(rgb_list=rgb_list, dpt_list=dpt_list, amp_list=amp_list, phs_list=phs_list, transform=transforms.Compose([transforms.ToTensor()]))
    #test_dataset=TensorHoloDataset(rgb_list=rgb_list, dpt_list=dpt_list, amp_list=amp_list, phs_list=phs_list, transform=None)
    test_loader=torch.utils.data.DataLoader(test_dataset, batch_size=1, shuffle=False)
    '''
    if os.path.isfile(CKPT_DIR+filename):
        print("Loading Checkpoint")
        checkpoint=torch.load(CKPT_DIR+filename)
        net.load_state_dict(checkpoint["model_state_dict"])
        optimizer.load_state_dict(checkpoint["optimizer_state_dict"])
    '''
    print('\n[ Test Epoch: %d ]'%epoch)
    
    net.eval()
    test_loss=0

    for batch_idx, (test_rgb, test_dpt, test_amp, test_phs) in enumerate(test_loader):
        test_rgb=test_rgb.to(device)
        test_dpt=test_dpt.to(device)
        test_amp=test_amp.to(device)
        test_phs=test_phs.to(device)

        output_amp, output_phs=net(test_rgb, test_dpt)
        loss_amp2=criterion(output_amp, test_amp)
        loss_phs2=criterion(output_phs, test_phs)
        loss2=loss_amp2+loss_phs2
        #delta=torch.atan2(torch.sin(output_phs-test_phs), torch.cos(output_phs-test_phs))
        #criterion1=output_amp-test_amp*torch.exp(1j*(delta-delta.conj()))
        #criterion1=torch.linalg.norm(criterion1, ord=2, axis=1)
        #loss=torch.sum(criterion1)
        
        #test_loss=loss.backward()
        test_loss+=loss2.item()
        output_amp = output_amp.detach().cpu().numpy()
        output_phs = output_phs.detach().cpu().numpy()

        output_amp=np.transpose(output_amp[0,:,:,:], [1,2,0])
        output_amp=cv2.cvtColor(output_amp, cv2.COLOR_BGR2RGB)
        output_phs=np.transpose(output_phs[0,:,:,:], [1,2,0])
        output_phs=cv2.cvtColor(output_phs, cv2.COLOR_BGR2RGB)

        cv2.imwrite(os.path.join(RESULTS_DIR, 'amp_%d.bmp'%(800+batch_idx)), output_amp*255.0)
        cv2.imwrite(os.path.join(RESULTS_DIR, 'phs_%d.bmp'%(800+batch_idx)), output_phs*255.0)
    print('Test Average Loss: ', test_loss) # 200 -> len(test data)
    state={
        'net':net.state_dict()
    }
    if not os.path.isdir('ckpt'):
        os.mkdir('ckpt')
    torch.save(state, './ckpt/'+filename)
    print('Model Saved!')

    

def adjust_learning_rate(optimizer, epoch):
    lr = learning_rate
    if epoch >= 100:
        lr /= 10
    if epoch >= 150:
        lr /= 10
    for param_group in optimizer.param_groups:
        param_group['lr'] = lr

if __name__=='__main__':
    
    for epoch in range(0, 1000):
        adjust_learning_rate(optimizer, epoch)
        train(epoch)
        test(epoch)
    