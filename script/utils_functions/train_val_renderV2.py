
import numpy as np
from tqdm import tqdm
import torch
import torch.nn as nn
import  pandas as pd
import matplotlib.pyplot as plt
from utils_functions.renderBatchItem import renderBatchSil
from utils_functions.testRender import testRenderResnet
from utils_functions.R2Rmat import R2Rmat



def train_renderV2(model, train_dataloader, test_dataloader,
                 n_epochs, loss_function,
                 date4File, cubeSetName, batch_size, fileExtension, device, obj_name, noise, number_train_im):
    # monitor loss functions as the training progresses
    lr = 0.0001

    loop = n_epochs
    Step_losses = []
    current_step_loss = []
    Epoch_losses = []
    count = 0
    renderCount = 0
    regressionCount = 0
    renderbar = []
    regressionbar = []
    numbOfImageDataset = number_train_im


    for epoch in tqdm(range(n_epochs)):

        ## Training phase
        model.train()
        print('train phase epoch {}'.format(epoch))

        for image, silhouette, parameter in train_dataloader:
            loss = 0
            Step_loss = 0
            numbOfImage = image.size()[0]
            image = image.to(device)
            parameter = parameter.to(device)
            silhouette = silhouette.to(device)

            params = model(image) #should be size [batchsize, 6]
            # print('computed parameters are {}'.format(params))
            # print(params.size())

            for i in range(0,numbOfImage):
                #create and store silhouette
                model.t = params[i, 3:6]
                R = params[i, 0:3]
                # print(R)
                # print(model.t)
                model.R = R2Rmat(R)  # angle from resnet are in radian
                # print(model.t)
                # print(model.R)
                current_sil = model.renderer(model.vertices, model.faces, R=model.R, t=model.t, mode='silhouettes').squeeze()
                # print(current_sil)
                current_GT_sil = (silhouette[i]/255).type(torch.FloatTensor).to(device)
                # print(current_GT_sil)
                if (model.t[2] > 1 and model.t[2] < 10 and torch.abs(model.t[0]) < 1.5 and torch.abs(model.t[1]) < 1.5):
                    optimizer = torch.optim.Adam(model.parameters(), lr=lr)
                    loss  +=  nn.BCELoss()(current_sil, current_GT_sil)
                    print('render')
                    renderCount += 1
                else:
                    optimizer = torch.optim.Adam(model.parameters(), lr=0.001)
                    loss  +=  nn.MSELoss()(params[i, 3:6], parameter[i, 3:6]).to(device)
                    print('regression')
                    regressionCount += 1

            loss = loss/numbOfImage #take the mean of the step loss


            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            print(loss)
            Step_losses.append(loss.detach().cpu().numpy()) # contain all step value for all epoch
            current_step_loss.append(loss.detach().cpu().numpy()) #contain only this epoch loss, will be reset after each epoch
            count = count+1

        epochloss = np.mean(current_step_loss)
        current_step_loss = []
        Epoch_losses.append(epochloss) #most significant value to store
        print(epochloss)
        print(renderCount, regressionCount)
        renderbar.append(renderCount)
        regressionbar.append(regressionCount)
        renderCount = 0
        regressionCount = 0

    fig, (p1, p11, p2, p3) = plt.subplots(4, figsize=(15,10)) #largeur hauteur

    #subplot 1
    rollingAv = pd.DataFrame(Step_losses)
    rollingAv.rolling(2).sum()
    p1.plot(np.arange(count), rollingAv, label="step Loss rolling average")
    p1.set( ylabel='BCE Step Loss')
    p1.set_ylim([0, 2])
    # Place a legend to the right of this smaller subplot.
    p1.legend()

    p11.plot(np.arange(count), Step_losses, label="step Loss")
    p11.set( ylabel='BCE Step Loss')
    p11.set_ylim([0, 2])
    # Place a legend to the right of this smaller subplot.
    p11.legend()

    #subplot 2
    p2.plot(np.arange(n_epochs), Epoch_losses, label="epoch Loss")
    p2.set( ylabel=' Mean of BCE step loss')
    p2.set_ylim([0, 2])
    # Place a legend to the right of this smaller subplot.
    p2.legend()

    #subplot 3
    ind = np.arange(n_epochs) #index
    width = 0.35
    p31 = plt.bar(ind, renderbar, width, color='#d62728')
    height_cumulative = renderbar
    p32 = plt.bar(ind, regressionbar, width, bottom=height_cumulative)

    plt.ylabel('render/regression call')
    plt.xlabel('epoch')
    p3.set_ylim([0, numbOfImageDataset])
    plt.xticks(ind)
    # Place a legend to the right of this smaller subplot.
    plt.legend((p31[0], p32[0]), ('render', 'regression'))
    plt.show()

    fig.savefig('results/render_{}batch_{}.pdf'.format(batch_size, n_epochs))
    import matplotlib2tikz

    matplotlib2tikz.save("results/render_{}batch_{}.tex".format(batch_size, n_epochs))


    #validation phase
    model.eval()


    for image, silhouette, parameter in test_dataloader:

        Test_Step_loss = []
        numbOfImage = image.size()[0]

        image = image.to(device)
        parameter = parameter.to(device)
        silhouette = silhouette.to(device)

        params = model(image)  # should be size [batchsize, 6]
        # print('computed parameters are {}'.format(params))
        # print(params.size())

        for i in range(0, numbOfImage):
            model.t = params[i, 3:6]
            R = params[i, 0:3]
            model.R = R2Rmat(R)  # angle from resnet are in radian
            current_sil = model.renderer(model.vertices, model.faces, R=model.R, t=model.t,mode='silhouettes').squeeze()
            current_GT_sil = (silhouette[i] / 255).type(torch.FloatTensor).to(device)

            loss = nn.BCELoss()(current_sil, current_GT_sil)
            Test_Step_loss.append(loss.detach().cpu().numpy())





