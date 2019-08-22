"""
script to train a resnet 50 network only with n epoch
Version 4
render is done during the computation beside the render
"""

import time
import torch
import torch.nn as nn
import numpy as np
from tqdm import tqdm
import  matplotlib
import matplotlib.pyplot as plt
plt.switch_backend('agg')
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


def mkdir_p(mypath):
    '''Creates a directory. equivalent to using mkdir -p on the command line'''

    from errno import EEXIST
    from os import makedirs,path

    try:
        makedirs(mypath)
    except OSError as exc: # Python >2.5
        if exc.errno == EEXIST and path.isdir(mypath):
            pass
        else: raise


def RolAv(list, window = 2):

    mylist = list
    # print(mylist)
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

n_epochs = 49
modelName = 'Ubelix_lr0_0001_15000_180_abcR_50_6param_{}epoch_082119_Ubelix_lr0_0001_15000_180_abcR_50_6param_Temp_train_cubes_wrist1im_Head_15000datasetRotationTranslationM15_15_5_7_8batchs_50epochs_Render'.format(n_epochs)


file_name_extension = 'wrist1im_Head_1000img_sequence_RotationTranslation180'  # choose the corresponding database to use

batch_size = 4



target_size = (512, 512)


cubes_file = 'Npydatabase/cubes_{}.npy'.format(file_name_extension)
silhouettes_file = 'Npydatabase/sils_{}.npy'.format(file_name_extension)
parameters_file = 'Npydatabase/params_{}.npy'.format(file_name_extension)
date4File = '082019' #mmddyy
fileExtension = '{}_TEST_RenderRotationTranslation_epoch{}Lr0_0001_55_6param'.format(date4File,n_epochs) #string to ad at the end of the file



obj_name = 'wrist'


cubes = np.load(cubes_file)
sils = np.load(silhouettes_file)
params = np.load(parameters_file)

#  ------------------------------------------------------------------

# test = 4
# test_im = cubes[:test]
# test_sil = sils[:test]
# test_param = params[:test]
# number_testn_im = np.shape(test_im)[0]


test_im = cubes
test_sil = sils
test_param = params
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

output_dir_model = 'models/render/{}'.format(fileExtension)
mkdir_p(output_dir_model)
output_dir_results = 'results/render/{}'.format(fileExtension)
mkdir_p(output_dir_results)

current_dir = os.path.dirname(os.path.realpath(__file__))
result_dir = os.path.join(current_dir, output_dir_results )
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
renderCount = 0
renderbar = []
renderbar = []

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

        # check some image
        if(processcount % 250 == 0):
            fig = plt.figure()
            plt.imshow(imgCP)
            plt.show()
            plt.close(fig)

        processcount = processcount+1
        # print(processcount)
        # print(params[i, 3])
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
    print(pickim)
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

plt.show()
plt.savefig('{}/image_render_{}batch_after{}epochs_{}.pdf'.format(output_dir_results, batch_size, n_epochs, fileExtension), bbox_inches = 'tight', pad_inches = 0.05)

## ----------- translation plot ------------------------------------------------------------------------------------------

fig, (px, py, pz) = plt.subplots(nrows=1, ncols=3, figsize=(9, 3))  # largeur hauteur
fig.suptitle("Render Model Test after {} epochs, Translation, Red = Ground Truth, Blue = Tracking".format(n_epochs), fontsize=6)

# TestCPparamX = RolAv(TestCPparamX, window=10)
px.plot(np.arange(np.shape(TestGTparamX)[0]), TestGTparamX, color = 'r', linestyle= '--')
px.plot(np.arange(np.shape(TestCPparamX)[0]), TestCPparamX, color = 'b')
px.set(ylabel='position [cm]')
px.set(xlabel='frame no.')
px.set_ylim([-2, 2])
px.set_title('Translation X')


# TestCPparamY = RolAv(TestCPparamY, window=10)
py.plot(np.arange(np.shape(TestGTparamY)[0]), TestGTparamY, color = 'r', linestyle= '--', label="Ground Truth")
py.plot(np.arange(np.shape(TestCPparamY)[0]), TestCPparamY, color = 'b', label="Ground Truth")
# py.set(ylabel='position [cm]')
py.set(xlabel='frame no.')
py.set_ylim([-2, 2])
py.set_title('Translation Y')

# TestCPparamZ = RolAv(TestCPparamZ, window=10)
pz.plot(np.arange(np.shape(TestGTparamZ)[0]), TestGTparamZ, color = 'r', linestyle= '--')
pz.plot(np.arange(np.shape(TestCPparamZ)[0]), TestCPparamZ, color = 'b')
# pz.set(ylabel='position [cm]')
pz.set(xlabel='frame no.')
pz.set_ylim([2, 8])
pz.set_title('Translation Z')

plt.show()
# fig.savefig('results/ResultSequenceRenderTest/RenderTestTranslation_{}.pdf'.format(fileExtension), bbox_inches = 'tight', pad_inches = 0.05)
fig.savefig('{}/RenderTestTranslation_{}.png'.format(output_dir_results,fileExtension), bbox_inches = 'tight', pad_inches = 0.05)
matplotlib2tikz.save("{}/RenderTestTranslation_{}.tex".format(output_dir_results,fileExtension),figureheight='5cm', figurewidth='5cm')

## ----------- rotation plot ------------------------------------------------------------------------------------------

fig2, (pa, pb, pg) = plt.subplots(nrows=1, ncols=3, figsize=(9, 3))  # largeur hauteur
fig.suptitle("Render Model Test after {} epochs, Rotation, Red = Ground Truth, Blue = Tracking".format(n_epochs), fontsize=14)

# TestCPparamA = TestCPparamA-np.mean(TestCPparamA) #use this if the rotation are a multiple of 2 pi
pa.plot(np.arange(np.shape(TestGTparamA)[0]), TestGTparamA, color = 'r', linestyle= '--')
pa.plot(np.arange(np.shape(TestCPparamA)[0]), TestCPparamA, color = 'b')
pa.set(ylabel='angle [rad]')
pa.set(xlabel='frame no.')
pa.set_ylim([-pi, 2*pi])
pa.set_title('Wrist Rotation')

# TestCPparamB = TestCPparamB-np.mean(TestCPparamB)
pb.plot(np.arange(np.shape(TestGTparamB)[0]), TestGTparamB, color = 'r', linestyle= '--')
pb.plot(np.arange(np.shape(TestCPparamB)[0]), TestCPparamB, color = 'b')
# pb.set(ylabel='angle [rad]')
pb.set(xlabel='frame no.')
pb.set_ylim([-pi, 2*pi])
pb.set_title('Wrist Rotation')

# TestCPparamC = TestCPparamC-np.mean(TestCPparamC)
pg.plot(np.arange(np.shape(TestGTparamC)[0]), TestGTparamC, color = 'r', linestyle= '--')
pg.plot(np.arange(np.shape(TestCPparamC)[0]), TestCPparamC, color = 'b')
# pg.set(ylabel='angle [rad]')
pg.set(xlabel='frame no.')
pg.set_ylim([-pi, 2*pi])
pg.set_title('Wrist Rotation')


plt.show()
# fig2.savefig('results/ResultSequenceRenderTest/RenderTestRotation_{}.pdf'.format(fileExtension), bbox_inches = 'tight', pad_inches = 0.05)
fig2.savefig('{}/RenderTestRotation_{}.png'.format(output_dir_results,fileExtension), bbox_inches = 'tight', pad_inches = 0.05)
matplotlib2tikz.save("{}/RenderTestRotation_{}.tex".format(output_dir_results,fileExtension),figureheight='5cm', figurewidth='5cm')


## ----------- error plot ------------------------------------------------------------------------------------------

fig3, (pxe, pye, pze) = plt.subplots(nrows=1, ncols=3, figsize=(9, 3))  # largeur hauteur

test = np.asarray(TestGTparamX)

# TestCPparamX = RolAv(TestCPparamX, window=10)
ErrorX = np.asarray(TestGTparamX)-np.asarray(TestCPparamX)
pxe.plot(np.arange(np.shape(TestCPparamX)[0]),np.abs(ErrorX), color = 'b')
pxe.set(ylabel='error [cm]')
pxe.set(xlabel='frame no.')
pxe.set_ylim([0, 1])
pxe.set_title('Error X')


# TestCPparamY = RolAv(TestCPparamY, window=10)
ErrorY = np.asarray(TestGTparamY)-np.asarray(TestCPparamY)
pye.plot(np.arange(np.shape(TestCPparamY)[0]), np.abs(ErrorY), color = 'b')
# py.set(ylabel='position [cm]')
pye.set(xlabel='error no.')
pye.set_ylim([0, 1])
pye.set_title('Error Y')

# TestCPparamZ = RolAv(TestCPparamZ, window=10)
ErrorZ = np.asarray(TestGTparamZ)-np.asarray(TestCPparamZ)
pze.plot(np.arange(np.shape(TestCPparamZ)[0]), np.abs(ErrorZ), color = 'b')
# pz.set(ylabel='position [cm]')
pze.set(xlabel='frame no.')
pze.set_ylim([0,2])
pze.set_title('Error Z')

plt.show()
# fig3.savefig('results/ResultSequenceRenderTest/RenderTestErrorTranslation_{}.pdf'.format(fileExtension), bbox_inches = 'tight', pad_inches = 0.05)
fig3.savefig('{}/RenderTestErrorTranslation_{}.png'.format(output_dir_results,fileExtension), bbox_inches = 'tight', pad_inches = 0.05)
matplotlib2tikz.save("{}/RenderTestErrorTranslation_{}.tex".format(output_dir_results,fileExtension),figureheight='5cm', figurewidth='5cm')


## ----------- error plot rotation------------------------------------------------------------------------------------------

fig4, (pae, pbe, pce) = plt.subplots(nrows=1, ncols=3, figsize=(9, 3))  # largeur hauteur

# test = np.asarray(TestGTparamX)

# TestCPparamX = RolAv(TestCPparamX, window=10)
ErrorA = np.asarray(TestGTparamA)-np.asarray(TestCPparamA)
pae.plot(np.arange(np.shape(TestCPparamA)[0]),np.abs(ErrorA), color = 'b')
pae.set(ylabel='error [rad]')
pae.set(xlabel='frame no.')
pae.set_ylim([0, pi])
pae.set_title('Err. Alpha')


# TestCPparamY = RolAv(TestCPparamY, window=10)
ErrorB = np.asarray(TestGTparamB)-np.asarray(TestCPparamB)
pbe.plot(np.arange(np.shape(TestCPparamB)[0]), np.abs(ErrorB), color = 'b')
# py.set(ylabel='position [cm]')
pbe.set(xlabel='frame no.')
pbe.set_ylim([0, pi])
pbe.set_title('Err. Beta')

# TestCPparamZ = RolAv(TestCPparamZ, window=10)
ErrorC = np.asarray(TestGTparamC)-np.asarray(TestCPparamC)
pce.plot(np.arange(np.shape(TestCPparamC)[0]), np.abs(ErrorC), color = 'b')
# pz.set(ylabel='position [cm]')
pce.set(xlabel='frame no.')
pce.set_ylim([0,pi])
pce.set_title('Err. Gamma')

plt.show()
# fig3.savefig('results/ResultSequenceRegressionTest/RegressionTestErrorTranslation_{}.pdf'.format(fileExtension), bbox_inches = 'tight', pad_inches = 0.05)
fig4.savefig('{}/RenderTestErrorRotation_{}.png'.format(output_dir_results,fileExtension), bbox_inches = 'tight', pad_inches = 0.05)
matplotlib2tikz.save("{}/RenderTestErrorRotation_{}.tex".format(output_dir_results, fileExtension),figureheight='5cm', figurewidth='5cm')

print("computing prediction done in  {} seconds ---".format(time.time() - start_time))




