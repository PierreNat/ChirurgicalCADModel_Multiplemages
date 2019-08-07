
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
    lr = 0.001

    loop = n_epochs
    Step_losses = []
    current_step_loss = []
    Test_losses = []
    Epoch_losses = []
    count = 0
    testcount = 0
    renderCount = 0
    regressionCount = 0
    renderbar = []
    regressionbar = []
    Im2ShowGT = []
    Im2ShowGCP = []
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




        #validation phase
        print('test phase epoch {}'.format(epoch))

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

                loss += nn.BCELoss()(current_sil, current_GT_sil)
                Test_Step_loss.append(loss.detach().cpu().numpy())
                if(epoch == n_epochs-1):

                    print('saving image to show')
                    imgCP, _, _ = model.renderer(model.vertices, model.faces, torch.tanh(model.textures), R=model.R,t=model.t)

                    imgCP= imgCP.squeeze()  # float32 from 0-1
                    imgCP = imgCP.detach().cpu().numpy().transpose((1, 2, 0))
                    imgCP = (imgCP * 255).astype(np.uint8)  # cast from float32 255.0 to 255 uint8
                    imgGT = image[i].detach().cpu().numpy()
                    imgGT = (imgGT * 0.5 + 0.5).transpose(1, 2, 0) #denormalization
                    Im2ShowGT.append(imgCP)
                    Im2ShowGCP.append(imgGT)

                    a = plt.subplot(2, numbOfImage, i + 1)
                    plt.imshow(imgGT)
                    a.set_title('GT {}'.format(i))
                    plt.xticks([0, 512])
                    plt.yticks([])
                    a = plt.subplot(2, numbOfImage, i + 1 + numbOfImage)
                    plt.imshow(imgCP)
                    a.set_title('Rdr {}'.format(i))
                    plt.xticks([0, 512])
                    plt.yticks([])


            loss = loss/numbOfImage
            Test_losses.append(loss.detach().cpu().numpy())
            loss = 0        # reset current test loss
            testcount = testcount+1



#-----------plot and save section ------------------------------------------------------------------------------------

    fig, (p1, p2, p3, p4) = plt.subplots(4, figsize=(15,10)) #largeur hauteur

    #subplot 1
    rollingAv = pd.DataFrame(Step_losses)
    rollingAv.rolling(2, win_type='triang').sum()
    p1.plot(np.arange(count), rollingAv, label="step Loss rolling average")
    p1.set( ylabel='BCE Step Loss')
    p1.set_ylim([0, 4])
    # Place a legend to the right of this smaller subplot.
    p1.legend()

    #subplot 2
    p2.plot(np.arange(n_epochs), Epoch_losses, label="epoch Loss")
    p2.set( ylabel=' Mean of BCE training step loss')
    p2.set_ylim([0, 4])
    # Place a legend to the right of this smaller subplot.
    p2.legend()

    #subplot 3
    ind = np.arange(n_epochs) #index
    width = 0.35
    p3.bar(ind, renderbar, width, color='#d62728', label="render")
    height_cumulative = renderbar
    p3.bar(ind, regressionbar, width, bottom=height_cumulative, label="regression")
    p3.set(ylabel='render/regression call')
    p3.set(xlabel='epoch')
    p3.set_ylim([0, numbOfImageDataset])
    p3.set_xticks(ind)
    p3.legend()

    p4.plot(np.arange(testcount), Test_losses, label="Test Loss")
    p4.set( ylabel='Mean of BCE test step loss')
    p4.set_ylim([0, 5])
    # Place a legend to the right of this smaller subplot.
    p4.legend()




    plt.show()

    fig.savefig('results/render_{}batch_{}.pdf'.format(batch_size, n_epochs))
    import matplotlib2tikz

    matplotlib2tikz.save("results/render_{}batch_{}.tex".format(batch_size, n_epochs))
