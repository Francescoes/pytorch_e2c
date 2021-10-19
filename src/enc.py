"""
Using the trained e2c:
    g- encode the images dataset and save the z dataset
    e- compute the eigenvalues for all the matrixes A(z) of the e2c (test in case
        the offset is 0)
    p- clusterize the samples in the latent space, plot and save the clusters
    (Kmean is used, other solutions could be considered as well, as GMM)
"""





import torch
import torchvision
import torch.optim as optim
import argparse
import matplotlib
import torch.nn as nn
import matplotlib.pyplot as plt
import torchvision.transforms as transforms
import model_pendulum
from tqdm import tqdm
from torchvision import datasets
from torch.utils.data import DataLoader
from torchvision.utils import save_image
matplotlib.style.use('ggplot')
import os
from PIL import Image
import numpy as np
import scipy.io
from numpy import linalg as LA


# import seaborn as sns; sns.set()  # for plot styling
from sklearn.cluster import KMeans
from sklearn.cluster import SpectralClustering




# construct the argument parser and parser the arguments
parser = argparse.ArgumentParser()
parser.add_argument('-g', '--gen', default=0, type=int,
                    help='do you want to generate the z dataset?')

parser.add_argument('-e', '--eig', default=0, type=int,
                    help='do you want to compute the eigenvalues for each A(z)?')

parser.add_argument('-p', '--plot', default=0, type=int,
                    help='do you want to plot the latent clusters and save them?')

args = vars(parser.parse_args())


generate_dataset = args['gen']
eigenvalues = args['eig']
plot = args['plot']


PATH = "./model/e2c_model_weights.pth"



class PendulumDataset(torch.utils.data.Dataset):
    def __init__(self, root="../input_pendulum/", transform=None, train = True):


        name =  "file_img.mat"

        mat = scipy.io.loadmat(root+name)

        # size_dataset = 5000
        y_all = mat['img'][:]
        u_all = mat['u'][:]



        IMG_dataset = []


        for idx in range(0, len(y_all)-1):
            img = np.array([1-(y_all[idx].reshape(40*40,)/255),1-(y_all[idx+1].reshape(40*40,)/255)], dtype='f')
            IMG_dataset.append(img)



        self.img = []

        for idx in range(len(IMG_dataset)-1):
            img2 = [IMG_dataset[idx],u_all[idx+1],IMG_dataset[idx+1]]
            self.img.append(img2)

        self.transform = transform

    def __len__(self):
        return len(self.img)

    def __getitem__(self, index):
        sample = self.img[index]

        if self.transform:
            sample1 = self.transform(sample[0])
            sample2 = self.transform(sample[2])
            u = (torch.from_numpy(sample[1])).float()
            sample = [sample1, u, sample2]

        return sample




if generate_dataset == 1:

    # learning parameters
    hist_len = 2
    batch_size = 64*hist_len
    lr = 0.0001
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')





    # transforms
    transform = transforms.Compose([
        transforms.ToTensor(),
    ])


    # train and validation data
    data = PendulumDataset(
        train=True,
        transform=transform
    )
    # print(train_data[0])
    # exit()


    # training and validation data loaders
    dataloader = DataLoader(
        data,
        batch_size=1,
        shuffle=False
    )




    model = model_pendulum.e2c().to(device)
    model.load_state_dict(torch.load(PATH))
    model.eval()




    model.eval()
    dataset_z = []
    samples_z = []

    with torch.no_grad():
        for i, data in tqdm(enumerate(dataloader), total=int(len(data)/dataloader.batch_size)):
            data = [elem.to(device).view(elem.size(0), -1) for elem in data]
            x_hat, mu, log_var, z, x_hat_t1, mu_t1, log_var_t1, z_t1, z_hat_t1 = model(data)
            sample = np.random.normal(mu.cpu(), np.exp(log_var.cpu()))[0]
            # print(sample)
            # print(np.array(mu.cpu()[0]))
            # exit()
            dataset_z.append(np.array(mu.cpu()[0],np.double))
            samples_z.append(sample)






    print("Dataset z size: ",len(dataset_z))

    mdic = {"z": dataset_z, "samples": samples_z}

    root = "dataset_z/"
    filename = "z_e2c.mat"
    scipy.io.savemat(root+filename, mdic)





if eigenvalues == 1:

    # learning parameters
    hist_len = 2
    batch_size = 64*hist_len
    lr = 0.0001
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')





    # transforms
    transform = transforms.Compose([
        transforms.ToTensor(),
    ])

    # train and validation data
    data = PendulumDataset(
        train=True,
        transform=transform
    )

    # training and validation data loaders
    dataloader = DataLoader(
        data,
        batch_size=1,
        shuffle=False
    )



    model = model_pendulum.e2c().to(device)
    model.load_state_dict(torch.load(PATH))
    model.eval()




    fig, (ax1, ax2, ax3) = plt.subplots(3, 1, figsize=(15,15))

    # plt.clf()
    # for stopping simulation with the esc key.
    plt.gcf().canvas.mpl_connect(
        'key_release_event',
        lambda event: [exit(0) if event.key == 'escape' else None])

    filename = "z_e2c.mat"
    root = "dataset_z/"
    mat = scipy.io.loadmat(root+filename)

    dataset_z = mat['z']
    samples_z = mat['samples']

    for k in range(int(len(data)/50)):
        element = data[k*50]
        element = [elem.to(device).view(elem.size(0), -1) for elem in element]
        x_hat, mu, log_var, z, x_hat_t1, mu_t1, log_var_t1, z_t1, z_hat_t1 = model(element)
        A, B, o = model.matrixes(z)
        features = 2
        A = A.view(features,features)
        B = B.view(features,1)
        # o = o.view(features)

        w, v = LA.eig(np.array(A.cpu().detach().numpy(),np.double))


        print(w)
        # print(element[0].shape)
        # exit()


        t = np.linspace(0,np.pi*2,100)
        circ = np.concatenate((np.cos(t),np.sin(t)))







        fig.suptitle('Title')

        ax1.clear()
        ax1.plot(np.cos(t), np.sin(t))
        ax1.plot(np.real(w[0]),np.imag(w[0]),"x")
        ax1.plot(np.real(w[1]),np.imag(w[1]),"x")
        ax1.set_xlim([-20, 20])
        ax1.set_ylim([-5, 5])
        plt.pause(.001)

        WIDTH, HEIGHT = 500,500
        im = Image.fromarray(np.array((element[0].view(2,40,40)[0]).cpu().detach().numpy(),np.double) * 255).resize((WIDTH, HEIGHT))
        # im.show()
        ax2.imshow(im)





        ax3.clear()
        plt.plot(dataset_z[:,0], dataset_z[:,1])
        marker = 'x'
        plt.plot(samples_z[:,0], samples_z[:,1], marker,label="marker='{0}'".format(marker))
        # print(mu)
        ax3.plot(mu[0][0].cpu().detach().numpy(), mu[0][1].cpu().detach().numpy(),'r^')
















if plot == 1:

    filename = "z_e2c.mat"
    root = "dataset_z/"
    mat = scipy.io.loadmat(root+filename)
    dataset_z = mat['z']
    samples_z = mat['samples']



    plt.figure(figsize=(15,15))
    # for marker in ['o', '.', ',', 'x', '+', 'v', '^', '<', '>', 's', 'd']:
    marker = '^'
    plt.plot(dataset_z[:,0], dataset_z[:,1])
    marker = 'v'
    plt.plot(samples_z[:,0], samples_z[:,1], marker,label="marker='{0}'".format(marker))
    # plt.legend(numpoints=1)

    plt.xlabel("z1")
    plt.ylabel("z2")

    plt.savefig('../outputs_pendulum/phase_planes/z_e2c.png')
    plt.show()






    root="../input_pendulum/"

    name =  "file_img.mat"



    mat = scipy.io.loadmat(root+name)



    size_dataset = 5000

    dataset_x = mat["x"][:]

    # print(dataset_x.shape)


    plt.figure(figsize=(15,15))
    # marker = "o"
    # plt.plot(dataset_x[:,0], dataset_x[:,1], marker,label="marker='{0}'".format(marker))
    plt.plot(dataset_x[:,0], dataset_x[:,1])
    # plt.legend(numpoints=1)

    plt.xlabel("x1")
    plt.ylabel("x2")



    plt.savefig('../outputs_pendulum/phase_planes/x_e2c.png')

    plt.show()


    x_min = int(min(samples_z[:, 0])-2)
    y_min = int(min(samples_z[:, 1])-2)
    x_max = int(max(samples_z[:, 0])+2)
    y_max = int(max(samples_z[:, 1])+2)

    kmeans = KMeans(n_clusters=2, random_state=0).fit(samples_z)



    plt.figure(figsize=(15,15))
    labels = kmeans.predict(samples_z)
    plt.scatter(samples_z[:, 0], samples_z[:, 1], marker = "^", c=labels, s=50, cmap='viridis')

    label1 = 0
    label2 = 1
    centers = kmeans.cluster_centers_ # cluster_centers_ndarray of shape (n_clusters, n_features)
    plt.scatter(centers[label1, 0], centers[label1, 1], c='red', s=400, alpha=0.5)
    plt.scatter(centers[label2, 0], centers[label2, 1], c='blue', s=400, alpha=0.5)

    point1 = centers[label1, :]
    point2 = centers[label2, :]

    middle_point = (point1+point2)/2
    m_ort = - (point2[0] -point1[0])/(point2[1] -point1[1])
    q_ort = middle_point[1]-m_ort*middle_point[0]

    plt.scatter(middle_point[0], middle_point[1], c='black', s=400, alpha=0.5);

    # x = np.linspace(x_min, x_max, 1000)
    # plt.plot(x, m_ort*x + q_ort, linestyle='solid')

    plt.savefig('../outputs_pendulum/clusters/clusters.png')
    plt.show()




    # print(m_ort,q_ort)
    if m_ort*centers[0, 0] + q_ort>centers[0, 1]:
        # cluster 0 under the line
        # cluster 1 above the line
        mdic = {"m": m_ort, "q": q_ort, "x_min": x_min, "x_max": x_max, "y_min": y_min, "y_max": y_max, "under": 0}
        print("point1 under the line")
    else:
        # cluster 0 above the line
        # cluster 1 under the line
        mdic = {"m": m_ort, "q": q_ort, "x_min": x_min, "x_max": x_max, "y_min": y_min, "y_max": y_max, "under": 1}
        print("point1 above the line")

    root = "./clusters/"
    filename = "line_coef.mat"


    scipy.io.savemat(root+filename, mdic)




    # # k-means is limited to linear cluster boundaries´
    # # The fundamental model assumptions of k-means (points will be closer to their
    # # own cluster center than to others) means that the algorithm will often be
    # # ineffective if the clusters have complicated geometries.
    # # In particular, the boundaries between k-means clusters will always be linear,
    # # which means that it will fail for more complicated boundaries.
    # # SpectralClustering estimator uses the graph of nearest neighbors to compute
    # # a higher-dimensional representation of the data, and then assigns labels
    # # using a k-means algorithm
    #
    #
    # model = SpectralClustering(n_clusters=2, affinity='nearest_neighbors',
    #                        assign_labels='kmeans')
    # labels = model.fit_predict(samples_z)
    #
    # plt.figure(figsize=(15,15))
    #
    # plt.scatter(samples_z[:, 0], samples_z[:, 1], marker = "^", c=labels, s=50, cmap='viridis');
    #
    #
    # plt.show()


    name =  "file_img.mat"
    root="/home/francesco/vae/input_pendulum/"
    mat = scipy.io.loadmat(root+name)

    # size_dataset = 5000
    u_all = mat['u'][:]


    first_cluster = []
    second_cluster = []
    first_cluster_t1 = []
    second_cluster_t1 = []
    u_first_cluster = []
    u_second_cluster = []
    for k in range(len(samples_z)-1):

        if labels[k]==label1:
            first_cluster.append(samples_z[k]-point1)
            first_cluster_t1.append(samples_z[k+1]-point1)
            u_first_cluster.append(u_all[k+1])
        elif labels[k]==label2:
            second_cluster.append(samples_z[k]-point2)
            second_cluster_t1.append(samples_z[k+1]-point2)
            u_second_cluster.append(u_all[k+1])

    first_cluster = np.array(first_cluster)
    second_cluster = np.array(second_cluster)

    u_first_cluster = np.array(u_first_cluster)
    u_second_cluster = np.array(u_second_cluster)


    # # Check what I saved
    # plt.figure(figsize=(15,15))
    # # for marker in ['o', '.', ',', 'x', '+', 'v', '^', '<', '>', 's', 'd']:
    # marker = '^'
    # plt.plot(first_cluster[:,0], first_cluster[:,1], marker,label="marker='{0}'".format(marker))
    # marker = 'v'
    # plt.plot(second_cluster[:,0], second_cluster[:,1], marker,label="marker='{0}'".format(marker))
    # plt.legend(numpoints=1)
    # plt.show()



    mdic = {"first_cluster": first_cluster, "second_cluster": second_cluster,
    "first_cluster_t1": first_cluster_t1, "second_cluster_t1": second_cluster_t1,
    "u_first_cluster": u_first_cluster, "u_second_cluster": u_second_cluster,
    "point1": point1, "point2": point2}


    root = "./clusters/"
    filename = "clusters.mat"


    scipy.io.savemat(root+filename, mdic)






    # from sklearn.mixture import GaussianMixture
    # gmm = GaussianMixture(n_components=2).fit(samples_z)
    #
    #
    # labels = gmm.predict(samples_z)
    #
    #
    # plt.figure(figsize=(15,15))
    #
    # plt.scatter(samples_z[:, 0], samples_z[:, 1], marker = "^", c=labels, s=50, cmap='viridis')
    #
    # # plt.show()
    # #
    # # plt.figure(figsize=(15,15))
    # z1, z2 = np.meshgrid(np.linspace(-5, 25), np.linspace(-5,25))
    # zz = np.array([z1.ravel(), z2.ravel()]).T
    # h = gmm.score_samples(zz)
    # h = h.reshape((50,50))
    # plt.contour(z1, z2, h)
    # plt.show()
    #
    # predic = gmm.predict_proba(zz)




    # # k-means is limited to linear cluster boundaries´
    # # The fundamental model assumptions of k-means (points will be closer to their
    # # own cluster center than to others) means that the algorithm will often be
    # # ineffective if the clusters have complicated geometries.
    # # In particular, the boundaries between k-means clusters will always be linear,
    # # which means that it will fail for more complicated boundaries.
    # # SpectralClustering estimator uses the graph of nearest neighbors to compute
    # # a higher-dimensional representation of the data, and then assigns labels
    # # using a k-means algorithm
    #
    #
    # model = SpectralClustering(n_clusters=2, affinity='nearest_neighbors',
    #                        assign_labels='kmeans')
    # labels = model.fit_predict(samples_z)
    #
    # plt.figure(figsize=(15,15))
    #
    # plt.scatter(samples_z[:, 0], samples_z[:, 1], marker = "^", c=labels, s=50, cmap='viridis');
    #
    #
    # plt.show()
