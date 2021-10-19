"""
Dynamics for the clusters: identify the matrixes (A,B) for the two clusters,
using the samples obtained encoding the images with the trained e2c
"""


import torch
import torchvision
import torch.optim as optim
import argparse
import matplotlib
import torch.nn as nn
import matplotlib.pyplot as plt
import torchvision.transforms as transforms
import numpy as np
import scipy.io
from tqdm import tqdm
from torchvision import datasets
from torch.utils.data import DataLoader
from torchvision.utils import save_image


from numpy import linalg as LA


# construct the argument parser and parser the arguments
parser = argparse.ArgumentParser()
parser.add_argument('-c', '--cluster', default=0, type=int,
                    help='id of the cluster')

args = vars(parser.parse_args())
cluster_num = args['cluster']



root = "./clusters/"
filename = "clusters.mat"

mat = scipy.io.loadmat(root+filename)

point1 = mat['point1']
point2 = mat['point2']


if cluster_num == 0:
    epoint = torch.tensor(point1).float().reshape(2,1)
    z = mat['first_cluster'][:]
    z_t1 = mat['first_cluster_t1'][:]
    u = mat['u_first_cluster'][:]
else:
    epoint = torch.tensor(point2).float().reshape(2,1)
    z = mat['second_cluster'][:]
    z_t1 = mat['second_cluster_t1'][:]
    u = mat['u_second_cluster'][:]





z_A = []
z_t1_A = []
for idx in range(len(z)-1):
    if (u[idx] == 0):
        z_A.append(z[idx])
        z_t1_A.append(z_t1[idx])

z_A = np.transpose(np.array(z_A))
z_t1_A = np.transpose(np.array(z_t1_A))



# z_t1 = A * z
A = np.matmul(z_t1_A,np.linalg.pinv(z_A))


print("A: ", A)
w, v = LA.eig(A)
print("As eigenvalues: ", w)





# Matrix B identification -> z_t1 = Az + Bu
# => z_t1 - Az = B u
# => B = (z_t1 - Az)pinv(u)

z = np.transpose(np.array(z))
u = np.transpose(np.array(u))
z_t1 = np.transpose(np.array(z_t1))



B = np.matmul(z_t1 - np.matmul(A,z),np.linalg.pinv(u))


print("B: ", B)

if cluster_num == 0:
    mdic = {"A0": A, "B0": B}


if cluster_num == 1:
    mdic = {"A1": A, "B1": B}


scipy.io.savemat("dataset_z/matrixes_"+str(cluster_num)+".mat", mdic)
