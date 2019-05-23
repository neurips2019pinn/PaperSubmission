import numpy as np
import torch
import torch.nn.functional as F
import matplotlib.pyplot as plt
import utils

def tensor_to_coord_tensor(X):
    """
    Takes in a 4D image and returns the 3*W*H coordinate representation
    """
    device = X.device
    batch_num, channel_num, height, width = X.shape
    flattened_img = X.permute((0, 1, 3, 2)).contiguous().view(batch_num, -1)
    x_coord = torch.arange(0, width).view(width, 1)
    x_coord = x_coord.expand(width, height).contiguous()
    x_coord = x_coord.view(width * height).float().expand(batch_num, -1).to(device)

    y_coord = torch.arange(height, 0, -1) - 1
    y_coord = y_coord.expand(width, height).contiguous()
    y_coord = y_coord.view(width * height).float().expand(batch_num, -1).to(device)
        
    coord_matrix = torch.stack([x_coord, y_coord, flattened_img])

    return coord_matrix.permute((1, 0, 2))


class KernelCompress(torch.nn.Module):
    def __init__(self, in_channels):
        super(KernelCompress, self).__init__()
        self.conv2d = torch.nn.Conv2d(in_channels, 1, 1)   # hidden layer
        self.conv2d.weight = torch.nn.Parameter(torch.ones(1,in_channels,1,1))
        self.conv2d.bias = torch.nn.Parameter(torch.zeros(1))
        self.softmax = torch.nn.Softmax2d()

    def forward(self, x):
        x = torch.abs(x)
        x = self.conv2d(x)
        x = x / torch.sum(self.conv2d.weight)
        return x

def rot_mat(theta):
    theta = torch.Tensor([theta])
    c, s = torch.cos(theta), torch.sin(theta)
    R = torch.Tensor(((c,-s), (s, c)))
    return R


class PoseNormalization(torch.nn.Module):
    def __init__(self, channel_num, img_shape):
        super(PoseNormalization, self).__init__()
        self.flip_y_loc = torch.Tensor([[2 , 0], [0, -2]])
        self.mu_R = torch.Tensor([0.5, 0.5])
        self.rotate_pi = torch.Tensor([[-1 ,0], [0, 1]])
        self.pi = 3.14159265358979323
        self.theta_R = (self.pi / 2)
        self.channel_compress = KernelCompress(channel_num)   # hidden layer
        # self.weights = torch.nn.Parameter(torch.eye(img_shape[1] * img_shape[0]), requires_grad=True)
        # self.weights = torch.nn.Parameter(torch.rand([img_shape[1] * img_shape[0], img_shape[1] * img_shape[0]]), requires_grad=True)
    
    def strength_cofficient(self, L1, L2):
        """
        Measure the relative strength of the first eigen value to the second
        """
        scaling_cofficient = 1
        eig_ratio = (L1 / L2) - 1
        if torch.nonzero(eig_ratio != eig_ratio).shape[0] > 0:
            print("BROKEN EIG VAL:", L1, L2)
            exception = Exception()
            exception.loc = torch.nonzero(eig_ratio != eig_ratio)[0][0]
            raise exception
        confidence = 2*(F.sigmoid(scaling_cofficient * eig_ratio) - 0.5)
        return confidence
    
    def left_svd_tensor(self, T):
        eps = 0.001
        device = T.device
        # X = torch.bmm(T, torch.bmm(self.weights.expand(T.shape[0], *self.weights.shape).to(device), T.permute(0, 2, 1)))
        X = torch.bmm(T, T.permute(0, 2, 1))

        D = X[:, 0, 0]* X[:, 1, 1] - X[:, 0, 1]*X[:, 1, 0]
        D = D.unsqueeze(1).unsqueeze(2)
        X = torch.mul((1 / (D + 0.001)), X)

        T = X[:, 0, 0] + X[:, 1, 1]
        D = X[:, 0, 0] * X[:, 1, 1] - X[:, 0, 1]*X[:, 1, 0]

        L1 = (T + torch.sqrt(F.relu((T**2) - 4*D))) / 2
        L2 = (T - torch.sqrt(F.relu((T**2) - 4*D))) / 2

        v1 = torch.stack([L1 - X[:, 1, 1], X[:, 1, 0]])
        v2 = torch.stack([L2 - X[:, 1, 1], X[:, 1, 0]])
        U = torch.stack([v1, v2]).permute(2, 0, 1)
        U = torch.nn.functional.normalize(U,  dim=2)
        confidence = self.strength_cofficient(L1, L2)
        return U, confidence, L1, L2

    def rot_mat_tensor(self, thetas):
        num_tensors = thetas.shape[0]
        c = torch.cos(thetas).unsqueeze(1).unsqueeze(2)
        s = torch.sin(thetas).unsqueeze(1).unsqueeze(2)
        R = torch.stack([torch.cat([c, -1*s], dim=1),  torch.cat([s, c], dim=1)], dim=1).squeeze(3)
        return R

    def tensor_orientation_transform(self, U, confidence):
        rot_lambda = torch.fmod(torch.atan2(U[:, 0, 1], U[:, 0, 0]) + 2 * self.pi, self.pi)
        diff_angle = (self.theta_R - rot_lambda)

        rotation = self.rot_mat_tensor(diff_angle).float()
        return rotation

    def forward(self, img_tensor):
        epsilon = 0.001
        # try:
        batch_size, channel_num, height, width = img_tensor.shape
        device = img_tensor.device
        img_scale_norm_mat = torch.Tensor([[1 / width , 0], [0, 1 / height]]).to(device)
        compressed_img_tensor = self.channel_compress(img_tensor)


        coord_tensor = tensor_to_coord_tensor(compressed_img_tensor)
        W = torch.abs(coord_tensor[:,2:3,:])
        X = coord_tensor[:, :2, :]
        W3 = torch.pow(W.clone(), 2) 

        WX = torch.mul(W3, X)

        mu_W = (torch.sum(WX, dim=2) / torch.sum(W3, dim=2) + epsilon).unsqueeze(2)

        XC = X - mu_W
        # print(XC.shape)
        # XC = torch.nn.functional.normalize(XC, dim=2)

        WXC = torch.mul(torch.sqrt(W), XC)
        WXC = WXC / WXC.sum()

        orientations, confidence, L1, L2 = self.left_svd_tensor(WXC)
        T = self.tensor_orientation_transform(orientations.clone(), confidence)


        mu_T = torch.mm(img_scale_norm_mat, mu_W.squeeze(2).t()).t() - self.mu_R.expand(X.shape[0], 2).to(device)
        mu_T = torch.mm(self.flip_y_loc.to(device), mu_T.t()).t().unsqueeze(2)

        theta = torch.cat([T, mu_T], dim=2)

        grid = F.affine_grid(theta, img_tensor.size())
        img_tensor = F.grid_sample(img_tensor, grid)
        # except Exception as exception:
        #     pass
        #     fig, ax = plt.subplots(2, 5, figsize=(50, 20))
        #     for index in range(10):
        #         plot_coords = (index // 5, index % 5)
        #         print(img_tensor[index:index+1].shape)
        #         ax[plot_coords[0], plot_coords[1]].imshow(utils.tensor_to_numpy_img(compressed_img_tensor[index:index+1]))
        #     plt.show()
        return img_tensor, mu_W, orientations, confidence, theta, (L1, L2)