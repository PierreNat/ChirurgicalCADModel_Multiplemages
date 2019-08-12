"""
script to train a resnet 50 network only with n epoch
Version 4
render is done during the computation beside the regression
"""

import time
import torch
import torch.nn as nn
import numpy as np
import tqdm
import  matplotlib
import matplotlib.pyplot as plt
from torch.utils.data import DataLoader
from torchvision.transforms import ToTensor, Compose, Normalize, Lambda
import os
import glob
import argparse
from skimage.io import imread, imsave
from utils_functions.MyResnet import Myresnet50
from utils_functions.render1item import render_1_image
from utils_functions.resnet50 import resnet50
from utils_functions.testRender import testRenderResnet
from utils_functions.cubeDataset import CubeDataset



# device = torch.device('cpu')
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
torch.cuda.empty_cache()
print(device)

modelName = '080919_Ubelix_Lr0_001BCE2take2_FinalModel_train_cubes_wrist1im_Head_10000dataset0_180_M2_2_5_8_8batchs_40epochs_Noise0.0_Ubelix_Lr0_001BCE2take2_Render'


file_name_extension = 'wrist1im_Head_1000img_sequence_Translation'  # choose the corresponding database to use

batch_size = 6

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

test_im = cubes
test_sil = sils
test_param = params

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
parser.add_argument('-or', '--filename_output', type=str,default=os.path.join(data_dir, 'ResultRender_{}.gif'.format(file_name_extension)))
parser.add_argument('-mr', '--make_reference_image', type=int, default=0)
parser.add_argument('-g', '--gpu', type=int, default=0)
args = parser.parse_args()

model = Myresnet50(filename_obj=args.filename_obj)

model.to(device)







model = resnet50(cifar=False, modelName=modelName) #train with the saved model from the training script
model = model.to(device)  # transfer the neural net onto the GPU
criterion = nn.MSELoss()

#  ------------------------------------------------------------------

# test the model
print("Start timer")
start_time = time.time()
parameters, predicted_params, test_losses, al, bl, gl, xl, yl, zl = testRenderResnet(model, test_dataloader, criterion, file_name_extension, device, obj_name)
print("computing prediction done in  {} seconds ---".format(time.time() - start_time))

#  ------------------------------------------------------------------
# display computed parameter against ground truth


obj_name = 'wrist'
ncols = 5
nrows = 2
Gt = []
Rdr = []
nb_im =5

fig = plt.figure()


# loop = tqdm.tqdm(range(0,nb_im))
for i in range(0, nb_im):
    randIm = i+1 #select a random image
    print('computed parameter_{}: '.format(i))
    print(predicted_params[randIm])
    print('ground truth parameter_{}: '.format(i))
    print(params[randIm])
    print('angle and translation error for {}: '.format(i))
    loss_angle = (predicted_params[randIm][0:3] - params[randIm][0:3])
    loss_translation = (predicted_params[randIm][3:6]-params[randIm][3:6])
    print(loss_angle, loss_translation)
    # print('error {} degree and {} meter '.format(np.rad2deg(predicted_params[randIm][0:3]-params[randIm][0:3]), predicted_params[randIm][3:6]-params[randIm][3:6]))


    im = render_1_image(obj_name, torch.from_numpy(predicted_params[randIm]))  # create the dataset

    Gt.append(test_im[randIm])
    Rdr.append(im)

    a = plt.subplot(2, nb_im, i+1)
    plt.imshow(test_im[randIm])
    a.set_title('GT {}'.format(i))
    plt.xticks([0, 500])
    plt.yticks([])
    a = plt.subplot(2, nb_im, i+1+nb_im)
    plt.imshow(im)
    a.set_title('Rdr {}'.format(i))
    plt.xticks([0, 500])
    plt.yticks([])

    # plt.subplot(2, nb_im, i+1)
    # plt.imshow(test_im[randIm])
    # plt.title('GT {}'.format(i))
    #
    # plt.subplot(2, nb_im, i+1+nb_im)
    # plt.imshow(im)
    # plt.title('Rdr {}'.format(i))


print('finish')


plt.subplots_adjust(left=None, bottom=None, right=None, top=None, wspace=0.1, hspace=None)
plt.tight_layout()
plt.savefig("image/GroundtruthVsRenderTestRt_realMultMovingbackground_rend2.png")
plt.close(fig)


