"""
script to train a resnet 50 network only with n epoch
Version 4
render is done during the computation beside the regression
"""

import time
import torch
import torch.nn as nn
import numpy as np
from tqdm import tqdm
import  matplotlib
import matplotlib.pyplot as plt
from torch.utils.data import DataLoader
from torchvision.transforms import ToTensor, Compose, Normalize, Lambda
import os
import glob
import argparse
import math
from utils_functions.R2Rmat import R2Rmat
from numpy.random import uniform
import matplotlib2tikz
from skimage.io import imread, imsave
from utils_functions.MyResnet import Myresnet50
import imageio
from skimage.io import imread, imsave
from utils_functions.cubeDataset import CubeDataset
pi = math.pi
def RolAv(list, window = 2):

    mylist = list
    print(mylist)
    N = window
    cumsum, moving_aves = [0], []

    for i, x in enumerate(mylist, 1):
        cumsum.append(cumsum[i - 1] + x)
        if i >= N:
            moving_ave = (cumsum[i] - cumsum[i - N]) / N
            # can do stuff with moving_ave here
            moving_aves.append(moving_ave)

    return moving_aves

def make_gif(filename):
    with imageio.get_writer(filename, mode='I') as writer:
        for filename in sorted(glob.glob('/tmp/_tmp_*.png')):
            writer.append_data(imread(filename))
            os.remove(filename)
    writer.close()

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
torch.cuda.empty_cache()
print(device)

modelName = '0epoch_080819_test_TempModel_train_cubes_wrist1im_Head_10000dataset_t_6batchs_1epochs_Noise0.0_test_RenderRegr'


file_name_extension = 'wrist1im_Head_1000img_sequence_Translation'  # choose the corresponding database to use

batch_size = 4

n_epochs = 1

target_size = (512, 512)


cubes_file = 'Npydatabase/cubes_{}.npy'.format(file_name_extension)
silhouettes_file = 'Npydatabase/sils_{}.npy'.format(file_name_extension)
parameters_file = 'Npydatabase/params_{}.npy'.format(file_name_extension)

fileExtension = 'test' #string to ad at the end of the file

date4File = '080819_{}'.format(fileExtension) #mmddyy

obj_name = 'wrist'


cubes = np.load(cubes_file)
sils = np.load(silhouettes_file)
params = np.load(parameters_file)

#  ------------------------------------------------------------------
test = 4
test_im = cubes[:test]
test_sil = sils[:test]
test_param = params[:test]
number_testn_im = np.shape(test_im)[0]

#  ------------------------------------------------------------------

normalize = Normalize(mean=[0.5], std=[0.5])
gray_to_rgb = Lambda(lambda x: x.repeat(3, 1, 1))
transforms = Compose([ToTensor(),  normalize])
test_dataset = CubeDataset(test_im, test_sil, test_param, transforms)

test_dataloader = DataLoader(test_dataset, batch_size=batch_size, shuffle=False, num_workers=2)

# for image, sil, param in test_dataloader:
#
#     nim = image.size()[0]
#     for i in range(0,nim):
#         print(image.size(), sil.size(), param.size()) #torch.Size([batch, 3, 512, 512]) torch.Size([batch, 6])
#         im = i
#         print(param[im])  # parameter in form tensor([2.5508, 0.0000, 0.0000, 0.0000, 0.0000, 5.0000])
#
#
#         image2show = image[im]  # indexing random  one image
#         print(image2show.size()) #torch.Size([3, 512, 512])
#         plt.imshow((image2show * 0.5 + 0.5).numpy().transpose(1, 2, 0))
#         plt.show()
#
#         image2show = sil[im]  # indexing random  one image
#         print(image2show.size())  # torch.Size([3, 512, 512])
#         image2show = image2show.numpy()
#         plt.imshow(image2show, cmap='gray')
#         plt.show()


#  ------------------------------------------------------------------
# Setup the model
current_dir = os.path.dirname(os.path.realpath(__file__))
result_dir = os.path.join(current_dir, 'results/ResultSequenceRenderTest')
data_dir = os.path.join(current_dir, 'data')

parser = argparse.ArgumentParser()
parser.add_argument('-io', '--filename_obj', type=str, default=os.path.join(data_dir, '{}.obj'.format(obj_name)))
parser.add_argument('-or', '--filename_output', type=str,default=os.path.join(result_dir, 'ResultRender_{}.gif'.format(file_name_extension)))
parser.add_argument('-mr', '--make_reference_image', type=int, default=0)
parser.add_argument('-g', '--gpu', type=int, default=0)
args = parser.parse_args()

model = Myresnet50(filename_obj=args.filename_obj,pretrained=True, cifar=False, modelName= modelName)
model.eval()
model.to(device)


#  ------------------------------------------------------------------

# test the model
print("Start timer")
start_time = time.time()

Step_Val_losses = []
current_step_loss = []
current_step_Test_loss = []
Test_losses = []
Epoch_Val_losses = []
images_losses = []
Epoch_Test_losses = []
count = 0
testcount = 0
Im2ShowGT = []
Im2ShowGCP = []

TestCPparam = []
TestGTparam = []
TestCPparamX = []
TestCPparamY = []
TestCPparamZ = []
TestCPparamA = []
TestCPparamB = []
TestCPparamC = []

TestGTparamX = []
TestGTparamY = []
TestGTparamZ = []
TestGTparamA = []
TestGTparamB = []
TestGTparamC = []
numbOfImageDataset = number_testn_im
processcount= 0
regressionCount = 0
renderbar = []
regressionbar = []

t = tqdm(iter(test_dataloader), leave=True, total=len(test_dataloader))
for image, silhouette, parameter in t:

    Test_Step_loss = []
    numbOfImage = image.size()[0]

    image = image.to(device)
    parameter = parameter.to(device)
    params = model(image)  # should be size [batchsize, 6]
    # print(np.shape(params))

    for i in range(0,numbOfImage):
        #create and store silhouette
        model.t = params[i, 3:6]
        R = params[i, 0:3]
        model.R = R2Rmat(R)  # angle from resnet are in radian

        current_sil = model.renderer(model.vertices, model.faces, R=model.R, t=model.t, mode='silhouettes').squeeze()
        current_GT_sil = (silhouette[i]/255).type(torch.FloatTensor).to(device)



        imgCP, _, _ = model.renderer(model.vertices, model.faces, torch.tanh(model.textures), R=model.R, t=model.t)
        imgGT = image[i]

        imgCP = imgCP.squeeze()  # float32 from 0-1
        imgCP = imgCP.detach().cpu().numpy().transpose((1, 2, 0))
        imgCP = (imgCP * 255).astype(np.uint8)  # cast from float32 255.0 to 255 uint8
        imgGT = imgGT.squeeze()  # float32 from 0-1
        imgGT = imgGT.detach().cpu().numpy().transpose((1, 2, 0))
        imgGT = (imgGT * 255).astype(np.uint8)  # cast from float32 255.0 to 255 uint8

        loss = nn.BCELoss()(current_sil, current_GT_sil).to(device)
        imsave('/tmp/_tmp_%04d.png' % processcount, imgCP)
        processcount = processcount+1
        print(processcount)
        print(params[i, 3])
        TestCPparamX.append(params[i, 3].detach().cpu().numpy())
        TestCPparamY.append(params[i, 4].detach().cpu().numpy())
        TestCPparamZ.append(params[i, 5].detach().cpu().numpy())
        TestCPparamA.append(params[i, 0].detach().cpu().numpy())
        TestCPparamB.append(params[i, 1].detach().cpu().numpy())
        TestCPparamC.append(params[i, 2].detach().cpu().numpy())

        TestGTparamX.append(parameter[i, 3].detach().cpu().numpy())
        TestGTparamY.append(parameter[i, 4].detach().cpu().numpy())
        TestGTparamZ.append(parameter[i, 5].detach().cpu().numpy())
        TestGTparamA.append(parameter[i, 0].detach().cpu().numpy())
        TestGTparamB.append(parameter[i, 1].detach().cpu().numpy())
        TestGTparamC.append(parameter[i, 2].detach().cpu().numpy())

    images_losses.append(loss.detach().cpu().numpy())


    # #save all parameters, computed and ground truth position
    TestCPparam.extend(params.detach().cpu().numpy()) #print(np.shape(TestCPparam)[0])
    TestGTparam.extend(parameter.detach().cpu().numpy())

    testcount = testcount + 1

make_gif(args.filename_output)



# ----------- plot some result from the last epoch computation ------------------------

    # print(np.shape(LastEpochTestCPparam)[0])
print(np.shape(TestGTparamX)[0])
nim = 5
for i in range(0, nim):
    print('saving image to show')
    pickim = int(uniform(0, testcount- 1))
    # print(pickim)

    model.t = torch.from_numpy(TestCPparam[pickim][3:6]).to(device)
    R = torch.from_numpy(TestCPparam[pickim][0:3]).to(device)
    model.R = R2Rmat(R)  # angle from resnet are in radia
    imgCP, _, _ = model.renderer(model.vertices, model.faces, torch.tanh(model.textures), R=model.R, t=model.t)

    model.t = torch.from_numpy(TestGTparam[pickim][3:6]).to(device)
    R = torch.from_numpy(TestGTparam[pickim][0:3]).to(device)
    model.R = R2Rmat(R)  # angle from resnet are in radia
    imgGT, _, _ = model.renderer(model.vertices, model.faces, torch.tanh(model.textures), R=model.R, t=model.t)

    imgCP = imgCP.squeeze()  # float32 from 0-1
    imgCP = imgCP.detach().cpu().numpy().transpose((1, 2, 0))
    imgCP = (imgCP * 255).astype(np.uint8)  # cast from float32 255.0 to 255 uint8
    imgGT = imgGT.squeeze()  # float32 from 0-1
    imgGT = imgGT.detach().cpu().numpy().transpose((1, 2, 0))
    imgGT = (imgGT * 255).astype(np.uint8)  # cast from float32 255.0 to 255 uint8
    Im2ShowGT.append(imgCP)
    Im2ShowGCP.append(imgGT)

    a = plt.subplot(2, nim, i + 1)
    plt.imshow(imgGT)
    a.set_title('GT {}'.format(i))
    plt.xticks([0, 512])
    plt.yticks([])
    a = plt.subplot(2, nim, i + 1 + nim)
    plt.imshow(imgCP)
    a.set_title('Rdr {}'.format(i))
    plt.xticks([0, 512])
    plt.yticks([])

plt.savefig('results/ResultSequenceRenderTest/image_render_{}batch_{}_{}.pdf'.format(batch_size, n_epochs, fileExtension))

# #----------- plot ground truth with computed for all 6 parameters -----------------------------------------------------

fig, (px, pz, py, pa, pb, pg) = plt.subplots(nrows=1, ncols=6, figsize=(15, 15))  # largeur hauteur
px.plot(np.arange(np.shape(TestGTparamX)[0]), TestGTparamX, color = 'g')
px.set(ylabel='position [cm]')
px.set(xlabel='frame no')
px.set_ylim([-1.5, 1.5])
px.set_title('Translation X')

py.plot(np.arange(np.shape(TestGTparamY)[0]), TestGTparamY, color = 'g')
py.set(ylabel='position [cm]')
py.set(xlabel='frame no')
py.set_ylim([-1.5, 1.5])
py.set_title('Translation Y')

pz.plot(np.arange(np.shape(TestGTparamZ)[0]), TestGTparamZ, color = 'g')
pz.set(ylabel='position [cm]')
pz.set(xlabel='frame no')
pz.set_ylim([-1.5, 1.5])
pz.set_title('Translation Z')

pa.plot(np.arange(np.shape(TestGTparamA)[0]), TestGTparamA, color = 'g')
pa.set(ylabel='angle [deg]')
pa.set(xlabel='frame no')
pa.set_ylim([- math.pi, pi])
pa.set_title('Wrist Alpha Rotation')

pb.plot(np.arange(np.shape(TestGTparamB)[0]), TestGTparamB, color = 'g')
pb.set(ylabel='angle [deg]')
pb.set(xlabel='frame no')
pb.set_ylim([- math.pi, pi])
pb.set_title('Wrist Beta Rotation')

pg.plot(np.arange(np.shape(TestGTparamC)[0]), TestGTparamC, color = 'g')
pg.set(ylabel='angle [deg]')
pg.set(xlabel='frame no')
pg.set_ylim([- math.pi, pi])
pg.set_title('Wrist Gamma Rotation')


plt.show()
# px.legend()  # Place a legend to the right of this smaller subplot.

# #-----------plot and save section ------------------------------------------------------------------------------------
#
# fig, (p1, p2, p3, p4) = plt.subplots(4, figsize=(15, 15))  # largeur hauteur
#
#
# moving_aves = RolAv(Step_Val_losses, window=40)
# ind = np.arange(n_epochs)  # index
#
# p1.plot(np.arange(np.shape(moving_aves)[0]), moving_aves, label="step Loss rolling average")
# p1.set(ylabel='BCE Step Loss')
# p1.set(xlabel='Steps')
# p1.set_ylim([0, 10])
# p1.legend()  # Place a legend to the right of this smaller subplot.
#
# # subplot 2
# p2.plot(np.arange(n_epochs), Epoch_Val_losses, label="Render epoch Loss")
# p2.set(ylabel=' Mean of BCE training step loss')
# p2.set(xlabel='Epochs')
# p2.set_ylim([0, 5])
# p2.set_xticks(ind)
# p2.legend()
#
# # subplot 3
#
# width = 0.35
# p3.bar(ind, renderbar, width, color='#d62728', label="render")
# height_cumulative = renderbar
# p3.bar(ind, regressionbar, width, bottom=height_cumulative, label="regression")
# p3.set(ylabel='render/regression call')
# p3.set(xlabel='Epochs')
# p3.set_ylim([0, numbOfImageDataset])
# p3.set_xticks(ind)
# p3.legend()
#
# # subplot 4
# p4.plot(np.arange(n_epochs), Epoch_Test_losses, label="Render Test Loss")
# p4.set(ylabel='Mean of BCE test step loss')
# p4.set(xlabel='Epochs')
# p4.set_ylim([0, 5])
# p4.legend()
#
#
# plt.show()
#
# fig.savefig('results/ResultSequenceRenderTest/render_{}batch_{}_{}.pdf'.format(batch_size, n_epochs, fileExtension))
#
#
# matplotlib2tikz.save("results/ResultSequenceRenderTest/render_{}batch_{}_{}.tex".format(batch_size, n_epochs, fileExtension))
#
#


print("computing prediction done in  {} seconds ---".format(time.time() - start_time))




