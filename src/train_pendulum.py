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




# construct the argument parser and parser the arguments
parser = argparse.ArgumentParser()
parser.add_argument('-e', '--epochs', default=10, type=int,
                    help='number of epochs to train the VAE for')


args = vars(parser.parse_args())




PATH = "./model/e2c_model_weights.pth"




class PendulumDataset(torch.utils.data.Dataset):
    def __init__(self, root="../input_pendulum/", transform=None, train = True):




        name =  "file_img.mat"

        mat = scipy.io.loadmat(root+name)

        # size_dataset = 5000
        y_all = mat['img'][:]
        u_all = mat['u'][:]


        IMG_dataset = []

        if train:
            t = range(0, int(.9*len(y_all))-1)
        else:
            t = range(int(.9*len(y_all)),len(y_all)-1)


        for idx in t:
            img = np.array([1-(y_all[idx].reshape(40*40,)/255),1-(y_all[idx+1].reshape(40*40,)/255)])
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
            sample1 = self.transform(sample[0]).float()
            sample2 = self.transform(sample[2]).float()
            u = (torch.from_numpy(sample[1])).float()
            sample = [sample1, u, sample2]

        return sample










# learning parameters

hist_len = 2

epochs = args['epochs']
batch_size = 64*4
lr = 0.0001
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')





# transforms
transform = transforms.Compose([
    transforms.ToTensor(),
])


# train and validation data
train_data = PendulumDataset(
    train=True,
    transform=transform
)
val_data = PendulumDataset(
    train=False,
    transform=transform
)





# print(train_data[0][0].shape)
# print(train_data[0][1].shape)
# print(train_data[0][2].shape)
# exit()






# training and validation data loaders
train_loader = DataLoader(
    train_data,
    batch_size=batch_size,
    shuffle=True
)
val_loader = DataLoader(
    val_data,
    batch_size=batch_size,
    shuffle=True
)






model = model_pendulum.e2c().to(device)



optimizer = optim.Adam(model.parameters(), lr=lr)
criterion = nn.BCELoss(reduction='sum')
criterion2 = nn.KLDivLoss(reduction='sum')







def final_loss(bce_loss, mu, logvar,bce_loss_t1, KLD_loss_z):
    """
    This function will add the reconstruction loss (BCELoss) and the
    KL-Divergence.
    KL-Divergence = 0.5 * sum(1 + log(sigma^2) - mu^2 - sigma^2)
    :param bce_loss: recontruction loss
    :param mu: the mean from the latent vector
    :param logvar: log variance from the latent vector
    """
    BCE = bce_loss
    BCE2 = bce_loss_t1
    KLDz = KLD_loss_z
    KLD = -0.5 * torch.sum(1 + logvar - mu.pow(2) - logvar.exp())

    # print( "BCE",BCE)
    # print( "BCE2",BCE2)
    # print( "KLDz",KLDz)
    # print( "KLD",KLD)
    return BCE + BCE2 + KLD/(100) + KLDz/(512)/(300)



def fit(model, dataloader):
    model.train()
    running_loss = 0.0
    for i, data in tqdm(enumerate(dataloader), total=int(len(train_data)/dataloader.batch_size)):
        data = [elem.to(device).view(elem.size(0), -1) for elem in data]
        optimizer.zero_grad()
        x_hat, mu, log_var, z, x_hat_t1, mu_t1, log_var_t1, z_t1, z_hat_t1 = model(data)
        bce_loss = criterion(x_hat.view(-1, 40*40), data[0].view(-1, 40*40))
        bce_loss_t1 = criterion(x_hat_t1.view(-1, 40*40), data[2].view(-1, 40*40))
        KLD_loss_z = criterion2(z_hat_t1, z_t1.detach())
        loss = final_loss(bce_loss, mu, log_var, bce_loss_t1, KLD_loss_z)
        running_loss += loss.item()
        loss.backward()
        optimizer.step()
    train_loss = running_loss/len(dataloader.dataset)
    return train_loss




def validate(model, dataloader):
    model.eval()
    running_loss = 0.0
    with torch.no_grad():
        for i, data in tqdm(enumerate(dataloader), total=int(len(val_data)/dataloader.batch_size)):
            data = [elem.to(device).view(elem.size(0), -1) for elem in data]
            optimizer.zero_grad()
            x_hat, mu, log_var, z, x_hat_t1, mu_t1, log_var_t1, z_t1, z_hat_t1 = model(data)
            bce_loss = criterion(x_hat.view(-1, 40*40), data[0].view(-1, 40*40))
            bce_loss_t1 = criterion(x_hat_t1.view(-1, 40*40), data[2].view(-1, 40*40))
            KLD_loss_z = criterion2(z_hat_t1, z_t1)
            loss = final_loss(bce_loss, mu, log_var, bce_loss_t1, KLD_loss_z)
            running_loss += loss.item()

            # save the last batch input and output of every epoch
            if i == int(len(val_data)/dataloader.batch_size) - 1:
                num_rows = 8
                both = torch.cat((data[0].view(batch_size*hist_len, 1, 40, 40)[:8],
                                  x_hat.view(batch_size*hist_len, 1, 40, 40)[:8]))
                save_image(both.cpu(), f"../outputs_pendulum/images/output{epoch}.png", nrow=num_rows)
    val_loss = running_loss/len(dataloader.dataset)
    return val_loss





train_loss = []
val_loss = []
for epoch in range(epochs):
    print(f"Epoch {epoch+1} of {epochs}")
    train_epoch_loss = fit(model, train_loader)
    val_epoch_loss = validate(model, val_loader)
    train_loss.append(train_epoch_loss)
    val_loss.append(val_epoch_loss)
    print(f"Train Loss: {train_epoch_loss:.4f}")
    print(f"Val Loss: {val_epoch_loss:.4f}")



plt.figure(figsize=(10,5))
plt.title("Training and Validation Loss")
plt.plot(val_loss,label="val")
plt.plot(train_loss,label="train")
plt.xlabel("Epochs")
plt.ylabel("Loss")
plt.legend()




plt.savefig('../outputs_pendulum/train_val_plot/Loss_e2c.png')


plt.show()


# Save model weights
torch.save(model.state_dict(), PATH)
