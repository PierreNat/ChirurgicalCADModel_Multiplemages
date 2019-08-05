from torch.autograd import Variable
import torch



def R2Rmat(R, n_comps=1):
    #function use to make the angle into matrix for the projection function of the renderer

    # R[0] = 1.0472
    # R[1] = 0
    # R[2] = 0.698132
    alpha = R[0] #already in radian
    beta = R[1]
    gamma =  R[2]

    rot_x = Variable(torch.zeros(n_comps, 3, 3).cuda(), requires_grad=False)
    rot_y = Variable(torch.zeros(n_comps, 3, 3).cuda(), requires_grad=False)
    rot_z = Variable(torch.zeros(n_comps, 3, 3).cuda(), requires_grad=False)
    rot_x[:, 0, 0] = 1
    rot_x[:, 0, 1] = 0
    rot_x[:, 0, 2] = 0
    rot_x[:, 1, 0] = 0
    rot_x[:, 1, 1] = alpha.cos()
    rot_x[:, 1, 2] = -alpha.sin()
    rot_x[:, 2, 0] = 0
    rot_x[:, 2, 1] = alpha.sin()
    rot_x[:, 2, 2] = alpha.cos()

    rot_y[:, 0, 0] = beta .cos()
    rot_y[:, 0, 1] = 0
    rot_y[:, 0, 2] = beta .sin()
    rot_y[:, 1, 0] = 0
    rot_y[:, 1, 1] = 1
    rot_y[:, 1, 2] = 0
    rot_y[:, 2, 0] = -beta .sin()
    rot_y[:, 2, 1] = 0
    rot_y[:, 2, 2] = beta.cos()

    rot_z[:, 0, 0] = gamma.cos()
    rot_z[:, 0, 1] = -gamma.sin()
    rot_z[:, 0, 2] = 0
    rot_z[:, 1, 0] = gamma.sin()
    rot_z[:, 1, 1] = gamma.cos()
    rot_z[:, 1, 2] = 0
    rot_z[:, 2, 0] = 0
    rot_z[:, 2, 1] = 0
    rot_z[:, 2, 2] = 1


    R = torch.bmm(rot_z, torch.bmm(rot_y, rot_x))
    # print(R)
    # cp_rotMat = (R)  # cp_rotMat = (model.R).detach().cpu().numpy()
    # r = Rot.from_dcm(cp_rotMat.detach().cpu().numpy())
    # r_euler = r.as_euler('xyz', degrees=True)
    # print('reuler: {}'.format(r_euler))
    return R