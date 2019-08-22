
import numpy as np
from tqdm import tqdm
import torch
import torch.nn as nn
import  pandas as pd
import matplotlib.pyplot as plt
from numpy.random import uniform

from utils_functions.R2Rmat import R2Rmat


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



def train_regressionV2(model, train_dataloader, test_dataloader,
                 n_epochs, loss_function,
                 date4File, cubeSetName, batch_size, fileExtension, device, obj_name, noise, number_train_im):
    # monitor loss functions as the training progresses
    lr = 0.001
    loop = n_epochs
    Step_Val_losses = []
    current_step_loss = []
    current_step_Test_loss = []
    Test_losses = []
    Epoch_Val_losses = []
    Epoch_Test_losses = []
    count = 0
    testcount = 0
    Im2ShowGT = []
    Im2ShowGCP = []
    LastEpochTestCPparam = []
    LastEpochTestGTparam = []
    numbOfImageDataset = number_train_im


    for epoch in range(n_epochs):

        ## Training phase
        model.train()
        print('train phase epoch {}/{}'.format(epoch, n_epochs))
        optimizer = torch.optim.Adam(model.parameters(), lr=0.001)

        t = tqdm(iter(train_dataloader), leave=True, total=len(train_dataloader))
        for image, silhouette, parameter in t:
            image = image.to(device)
            parameter = parameter.to(device)
            params = model(image)  # should be size [batchsize, 6]
            optimizer.zero_grad()
            numbOfImage = image.size()[0]
            # loss = nn.MSELoss()(params, parameter).to(device)

            for i in range(0,numbOfImage):
                #create and store silhouette
                if (i == 0):
                    loss =nn.MSELoss()(params[i], parameter[i]).to(device)
                else:
                    loss = loss + nn.MSELoss()(params[i], parameter[i]).to(device)


            loss.backward()
            optimizer.step()
            # print(loss)
            Step_Val_losses.append(loss.detach().cpu().numpy())  # contain all step value for all epoch
            current_step_loss.append(loss.detach().cpu().numpy())  # contain only this epoch loss, will be reset after each epoch
            count = count + 1

        epochValloss = np.mean(current_step_loss)
        current_step_loss = []
        Epoch_Val_losses.append(epochValloss)  # most significant value to store

        torch.save(model.state_dict(),
                   'models/{}_{}epoch_{}_TempModel_train_{}_{}batchs_{}epochs_Noise{}_Regression.pth'.format(fileExtension, epoch, date4File,
                                                                                                      cubeSetName,
                                                                                                      str(batch_size),
                                                                                                      str(n_epochs),
                                                                                                      noise * 100,
                                                                                                      ))
        # print(epochValloss)

        # validation phase
        print('test phase epoch epoch {}/{}'.format(epoch, n_epochs))
        model.eval()

        t = tqdm(iter(test_dataloader), leave=True, total=len(test_dataloader))
        for image, silhouette, parameter in t:

            Test_Step_loss = []
            numbOfImage = image.size()[0]

            image = image.to(device)
            parameter = parameter.to(device)
            params = model(image)  # should be size [batchsize, 6]
            # print(np.shape(params))

            for i in range(0,numbOfImage):
                if (i == 0):
                    loss =nn.MSELoss()(params[i], parameter[i]).to(device)
                else:
                    loss = loss + nn.MSELoss()(params[i], parameter[i]).to(device)

            Test_Step_loss.append(loss.detach().cpu().numpy())

            if (epoch == n_epochs - 1):  # if we are at the last epoch, save param to plot result

                LastEpochTestCPparam.extend(params.detach().cpu().numpy())
                LastEpochTestGTparam.extend(parameter.detach().cpu().numpy())

            Test_losses.append(loss.detach().cpu().numpy())
            current_step_Test_loss.append(loss.detach().cpu().numpy())
            testcount = testcount + 1

        epochTestloss = np.mean(current_step_Test_loss)
        current_step_Test_loss = []
        Epoch_Test_losses.append(epochTestloss)  # most significant value to store

    # ----------- plot some result from the last epoch computation ------------------------

    # print(np.shape(LastEpochTestCPparam)[0])
    nim = 5
    for i in range(0, nim):
        print('saving image to show')
        pickim = int(uniform(0, np.shape(LastEpochTestCPparam)[0] - 1))
        # print(pickim)

        model.t = torch.from_numpy(LastEpochTestCPparam[pickim][3:6]).to(device)
        R = torch.from_numpy(LastEpochTestCPparam[pickim][0:3]).to(device)
        model.R = R2Rmat(R)  # angle from resnet are in radia
        imgCP, _, _ = model.renderer(model.vertices, model.faces, torch.tanh(model.textures), R=model.R, t=model.t)

        model.t = torch.from_numpy(LastEpochTestGTparam[pickim][3:6]).to(device)
        R = torch.from_numpy(LastEpochTestGTparam[pickim][0:3]).to(device)
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

    plt.savefig('results/image_regression_{}batch_{}_{}.pdf'.format(batch_size, n_epochs, fileExtension))

    # -----------plot and save section ------------------------------------------------------------------------------------

    fig, (p1, p2, p4) = plt.subplots(3, figsize=(15, 10))  # largeur hauteur

    moving_aves = RolAv(Step_Val_losses, window=20)

    p1.plot(np.arange(np.shape(moving_aves)[0]), moving_aves, label="step Loss rolling average")
    p1.set(ylabel='BCE Step Loss')
    p1.set_yscale('log')
    p1.set(xlabel='Steps')
    p1.set_ylim([0, 4])
    p1.legend()  # Place a legend to the right of this smaller subplot.

    # subplot 2
    p2.plot(np.arange(n_epochs), Epoch_Val_losses, label="epoch Loss")
    p2.set(ylabel=' Mean of BCE training step loss')
    p2.set(xlabel='Epochs')
    p2.set_ylim([0, 4])
    p2.legend()

    p4.plot(np.arange(n_epochs), Epoch_Test_losses, label="Test Loss")
    p4.set(ylabel='Mean of BCE test step loss')
    p4.set(xlabel='Epochs')
    p4.set_ylim([0, 2])
    p4.legend()

    plt.show()

    fig.savefig('results/regression_{}batch_{}_{}.pdf'.format(batch_size, n_epochs, fileExtension))
    import matplotlib2tikz

    matplotlib2tikz.save("results/regression_{}batch_{}_{}.tex".format(batch_size, n_epochs, fileExtension))
