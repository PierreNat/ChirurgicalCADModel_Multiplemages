
"""
script to train a resnet 50 network only with n epoch

rendering directly after each parameter estimation
"""
import torch
import argparse
import os
import numpy as np
import matplotlib.pyplot as plt
import torch.nn as nn
from torch.utils.data import DataLoader
from torchvision.transforms import ToTensor, Compose, Normalize, Lambda
from utils_functions.MyResnet import Myresnet50
from utils_functions.train_val_renderV2 import train_renderV2
from utils_functions.cubeDataset import CubeDataset

# device = torch.device('cpu')
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
torch.cuda.empty_cache()
print(device)

file_name_extension = 'wrist1im_Body_20dataset'  # choose the corresponding database to use

batch_size = 4

n_epochs = 4

target_size = (512, 512)


cubes_file = 'Npydatabase/cubes_{}.npy'.format(file_name_extension)
silhouettes_file = 'Npydatabase/sils_{}.npy'.format(file_name_extension)
parameters_file = 'Npydatabase/params_{}.npy'.format(file_name_extension)

fileExtension = 'MultipleTestConvergence' #string to ad at the end of the file

cubeSetName = 'cubes_{}'.format(file_name_extension) #used to describe the document name

date4File = '080619_{}'.format(fileExtension) #mmddyy

obj_name = 'AllTool'


cubes = np.load(cubes_file)
sils = np.load(silhouettes_file)
params = np.load(parameters_file)

#  ------------------------------------------------------------------

ratio = 1  # 90%training 10%validation
split = int(len(cubes)*ratio)
test_length = 10

train_im = cubes[:split]  # 90% training
train_sil = sils[:split]
train_param = params[:split]
number_train_im = np.shape(train_im)[0]

# val_im = cubes[split:]  # remaining ratio for validation
# val_sil = sils[split:]
# val_param = params[split:]

val_im = cubes[:test_length]  # remaining ratio for validation
val_sil = sils[:test_length]
val_param = params[:test_length]

test_im = cubes[:test_length]
test_sil = sils[:test_length]
test_param = params[:test_length]
number_testn_im = np.shape(test_im)[0]


#  ------------------------------------------------------------------

normalize = Normalize(mean=[0.5], std=[0.5])
gray_to_rgb = Lambda(lambda x: x.repeat(3, 1, 1))
transforms = Compose([ToTensor(),  normalize])
train_dataset = CubeDataset(train_im, train_sil, train_param, transforms)
val_dataset = CubeDataset(val_im, val_sil, val_param, transforms)
test_dataset = CubeDataset(test_im, test_sil, test_param, transforms)


train_dataloader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True, num_workers=2)
val_dataloader = DataLoader(val_dataset, batch_size=batch_size, shuffle=True, num_workers=2)
test_dataloader = DataLoader(test_dataset, batch_size=number_testn_im, shuffle=False, num_workers=2)


# for image, sil, param in train_dataloader:
#
# #plot silhouette
#     print(image.size(), sil.size(), param.size()) #torch.Size([batch, 3, 512, 512]) torch.Size([batch, 6])
#     im = 0
#     print(param[im])  # parameter in form tensor([2.5508, 0.0000, 0.0000, 0.0000, 0.0000, 5.0000])
#
#     image2show = image[im]  # indexing random  one image
#     print(image2show.size()) #torch.Size([3, 512, 512])
#     plt.imshow((image2show * 0.5 + 0.5).numpy().transpose(1, 2, 0))
#     plt.show()
#
#     image2show = sil[im]  # indexing random  one image
#     print(image2show.size())  # torch.Size([3, 512, 512])
#     image2show = image2show.numpy()
#     plt.imshow(image2show, cmap='gray')
#     plt.show()
#
#     break  # break here just to show 1 batch of data

#  ------------------------------------------------------------------
# Setup the model

current_dir = os.path.dirname(os.path.realpath(__file__))
data_dir = os.path.join(current_dir, 'data')


noise = 0.0
parser = argparse.ArgumentParser()
parser.add_argument('-io', '--filename_obj', type=str, default=os.path.join(data_dir, '{}.obj'.format(obj_name)))
parser.add_argument('-ir', '--filename_ref', type=str,default=os.path.join(data_dir, 'wrist1im_BodyR2_ref.png'))  # image result to target
parser.add_argument('-in', '--filename_init', type=str, default=os.path.join(data_dir, 'example5_inT.png'))  # image to init resnet with regression
parser.add_argument('-or', '--filename_output', type=str,default=os.path.join(data_dir, 'example5_resultR_render_1.gif'))
parser.add_argument('-mr', '--make_reference_image', type=int, default=0)
parser.add_argument('-g', '--gpu', type=int, default=0)
args = parser.parse_args()

model = Myresnet50(filename_obj=args.filename_obj, filename_ref=args.filename_ref, filename_init=args.filename_init)

model.to(device)

model.train(True)
bool_first = True
lr = 0.001

criterion = nn.BCELoss()  #nn.BCELoss()   #nn.CrossEntropyLoss()  define the loss (MSE, Crossentropy, Binarycrossentropy)
#
#  ------------------------------------------------------------------

train_renderV2(model, train_dataloader, test_dataloader,
                                        n_epochs, criterion,
                                        date4File, cubeSetName, batch_size, fileExtension, device, obj_name, noise, number_train_im)

#  ------------------------------------------------------------------

torch.save(model.state_dict(), 'models/{}_FinalModel_train_{}_{}batchs_{}epochs_Noise{}_{}_RenderRegr.pth'.format(date4File, cubeSetName, str(batch_size), str(n_epochs), noise*100,fileExtension))
print('parameters saved')

#  ------------------------------------------------------------------
