"""
Using the trained e2c, decode the cluster centers and visualize the images
"""




import torch
import torchvision
import torch.optim as optim
import argparse
import matplotlib
import torch.nn as nn
import matplotlib.pyplot as plt
import torchvision.transforms as transforms
from tqdm import tqdm
from torchvision import datasets
from torch.utils.data import DataLoader
from torchvision.utils import save_image
matplotlib.style.use('ggplot')

import os
from PIL import Image
import numpy as np
import scipy.io





features = 2

class decoder(nn.Module):
    def __init__(self):
        super(decoder, self).__init__()
        self.hist_len = 2
        self.w = 40
        self.h = 40

        f_maps_1 = 64
        f_size_1 = 7
        f_maps_2 = 32
        f_size_2 = 5
        f_maps_3 = 16
        f_size_3 = 3
        enc_dims = 100


        # decoder
        Dec_Layers = []
        Dec_Layers += [nn.Linear(in_features=features, out_features=enc_dims)]
        Dec_Layers += [nn.ReLU()]
        Dec_Layers += [nn.Linear(in_features=enc_dims, out_features=32*9*9)]
        Dec_Layers += [nn.ReLU()]


        self.dec1 = nn.Sequential(*Dec_Layers)


        Dec_Layers = []
        Dec_Layers += [nn.UpsamplingNearest2d(scale_factor=2)]
        Dec_Layers += [nn.Conv2d(in_channels=f_maps_2, out_channels=f_maps_2, kernel_size=f_size_2)]
        Dec_Layers += [nn.ReLU()]
        Dec_Layers += [nn.UpsamplingNearest2d(scale_factor=2)]
        Dec_Layers += [nn.Conv2d(in_channels=f_maps_2, out_channels=f_maps_1, kernel_size=f_size_2)]
        Dec_Layers += [nn.ReLU()]
        Dec_Layers += [nn.UpsamplingNearest2d(scale_factor=2)]
        Dec_Layers += [nn.Conv2d(in_channels=f_maps_1, out_channels=self.hist_len, kernel_size=f_size_2+4)]
        Dec_Layers += [nn.Sigmoid()]
        # Dec_Layers += [nn.View()]


        self.dec2 = nn.Sequential(*Dec_Layers)



    def forward(self, z):

        # decoding
        reconstruction = self.dec1(z)
        reconstruction = torch.reshape(reconstruction,(reconstruction.size(0),32,9,9))
        reconstruction = self.dec2(reconstruction).view(-1, self.hist_len, self.w*self.h)

        return reconstruction





# learning parameters

hist_len = 2

batch_size = 64*hist_len
lr = 0.0001
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')





# transforms
transform = transforms.Compose([
    transforms.ToTensor(),
])









model = decoder().to(device)
PATH = "./model/e2c_model_weights.pth"

weights = torch.load(PATH)

entries_to_remove = ('enc1.0.weight', 'enc1.0.bias', 'enc1.3.weight', 'enc1.3.bias', 'enc2.0.weight', 'enc2.0.bias', 'enc2.2.weight', 'enc2.2.bias',
"A_matrix.weight", "A_matrix.bias", "B_matrix.weight", "B_matrix.bias", "o_matrix.weight", "o_matrix.bias")

for k in entries_to_remove:
    del weights[k]



# print(weights.keys())


model.load_state_dict(weights)
model.eval()


root = "./clusters/"
filename = "clusters.mat"


mat = scipy.io.loadmat(root+filename)

point1 = mat["point1"][0].astype(np.float32)
point2 = mat["point2"][0].astype(np.float32)

data = torch.tensor(point1).reshape(1,-1).to(device)
reconstruction = model(data).cpu()



# print(reconstruction.reshape(2,40,40).shape)

img1 = reconstruction.reshape(2,40,40)[0].detach().numpy()


plt.imshow(img1,cmap=plt.get_cmap("gray"))
plt.show()




data = torch.tensor(point2).reshape(1,-1).to(device)
reconstruction = model(data).cpu()



# print(reconstruction.reshape(2,40,40).shape)

img1 = reconstruction.reshape(2,40,40)[0].detach().numpy()


plt.imshow(img1,cmap=plt.get_cmap("gray"))
plt.show()
