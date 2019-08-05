
import numpy as np
from tqdm import tqdm
import torch
import torch.nn as nn
import matplotlib.pyplot as plt
from utils_functions.renderBatchItem import renderBatchSil
from utils_functions.testRender import testRenderResnet
from utils_functions.R2Rmat import R2Rmat



def train_renderV2(model, train_dataloader, test_dataloader,
                 n_epochs, loss_function,
                 date4File, cubeSetName, batch_size, fileExtension, device, obj_name, noise):
    # monitor loss functions as the training progresses
    lr = 0.001

    loop = n_epochs
    Step_losses = []
    Epoch_losses = []
    count = 0
    for epoch in tqdm(range(n_epochs)):

        ## Training phase
        model.train()
        print('train phase epoch {}'.format(epoch))

        for image, silhouette, parameter in train_dataloader:
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
                    loss = nn.BCELoss()(current_sil, current_GT_sil)
                    print('render')
                else:
                    optimizer = torch.optim.Adam(model.parameters(), lr=0.001)
                    loss = nn.MSELoss()(params[i, 3:6], parameter[i, 3:6]).to(device)
                    print('regression')
                Step_loss = Step_loss + loss


            optimizer.zero_grad()
            Step_loss.backward()
            optimizer.step()
            print(Step_loss)
            Step_losses.append(Step_loss.detach().cpu().numpy())
            count = count+1

    fig, (p1, p2, p3) = plt.subplots(3, figsize=(15,10)) #largeur hauteur

    p1.plot(np.arange(count), Step_losses, label="Global Loss")
    p1.set( ylabel='BCE Loss')
    p1.set_ylim([0, 20])
    # Place a legend to the right of this smaller subplot.
    p1.legend()

    plt.show()

