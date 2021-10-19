import torch
import torch.nn as nn
import torch.nn.functional as F





features = 2


# define a VAE
class e2c(nn.Module):
    def __init__(self):
        super(e2c, self).__init__()
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

        # encoder
        Enc_Layers = []
        Enc_Layers += [nn.Conv2d(in_channels=self.hist_len, out_channels=f_maps_1, kernel_size=f_size_1)]
        Enc_Layers += [nn.ReLU()]
        Enc_Layers += [nn.MaxPool2d(2,2)]
        Enc_Layers += [nn.Conv2d(in_channels=f_maps_1, out_channels=f_maps_2, kernel_size=f_size_2)]
        Enc_Layers += [nn.ReLU()]
        Enc_Layers += [nn.MaxPool2d(2,2)]

        self.enc1 = nn.Sequential(*Enc_Layers)

        Enc_Layers = []
        Enc_Layers += [nn.Linear(in_features=32*6*6, out_features=512)]
        Enc_Layers += [nn.ReLU()]
        Enc_Layers += [nn.Linear(in_features=512, out_features=features*2)]
        Enc_Layers += [nn.ReLU()]
        self.enc2 = nn.Sequential(*Enc_Layers)

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




        self.A_matrix = nn.Linear(in_features=features, out_features=features*features,bias=True)
        self.B_matrix = nn.Linear(in_features=features, out_features=features,bias=True)
        self.o_matrix = nn.Linear(in_features=features, out_features=features,bias=True)




    def reparameterize(self, mu, log_var):
        """
        :param mu: mean from the encoder's latent space
        :param log_var: log variance from the encoder's latent space
        """
        std = torch.exp(0.5*log_var) # standard deviation
        eps = torch.randn_like(std) # `randn_like` as we need the same size
        sample = mu + (eps * std) # sampling as if coming from the input space
        return sample

    def forward_single(self, x):
        # encoding
        x = torch.reshape(x,(x.size(0), self.hist_len, self.w, self.h))
        x = self.enc1(x)
        x = torch.reshape(x,(x.size(0),32*6*6))
        x = self.enc2(x).view(-1, 2, features)




        # get `mu` and `log_var`
        mu = x[:, 0, :] # the first feature values as mean
        log_var = x[:, 1, :] # the other feature values as variance
        # get the latent vector through reparameterization
        z = self.reparameterize(mu, log_var)




        # decoding
        reconstruction = self.dec1(z)
        reconstruction = torch.reshape(reconstruction,(reconstruction.size(0),32,9,9))
        reconstruction = self.dec2(reconstruction).view(-1, self.hist_len, self.w*self.h)

        return reconstruction, mu, log_var, z

    def forward(self, input):

        x = input[0]
        u = input[1]
        x_t1 = input[2]

        x_hat, mu, log_var, z = self.forward_single(x)

        x_hat_t1, mu_t1, log_var_t1, z_t1 = self.forward_single(x_t1)


        A = self.A_matrix(z)
        B = self.B_matrix(z)
        o = self.o_matrix(z)

        # print(A.view(-1,features*features).size())
        # print(torch.matmul(B.view(-1,features,1),u.view(-1,1,1)).shape)
        # exit()
        z_hat_t1 = (torch.matmul(A.view(-1,features,features),z.view(-1,features,1))).view(-1,features) + torch.matmul(B.view(-1,features,1),u.view(-1,1,1)).view(-1,features) + o.view(-1,features)




        return x_hat, mu, log_var, z, x_hat_t1, mu_t1, log_var_t1, z_t1, z_hat_t1

    def matrixes(self, z):



        A = self.A_matrix(z)
        B = self.B_matrix(z)
        # o = self.o_matrix(z)




        return A, B, 1





# define a simple linear VAE
class LinearVAE(nn.Module):
    def __init__(self):
        super(LinearVAE, self).__init__()

        # encoder
        self.enc1 = nn.Linear(in_features=1600, out_features=512)
        self.enc2 = nn.Linear(in_features=512, out_features=features*2)

        # decoder
        self.dec1 = nn.Linear(in_features=features, out_features=512)
        self.dec2 = nn.Linear(in_features=512, out_features=1600)
    def reparameterize(self, mu, log_var):
        """
        :param mu: mean from the encoder's latent space
        :param log_var: log variance from the encoder's latent space
        """
        std = torch.exp(0.5*log_var) # standard deviation
        eps = torch.randn_like(std) # `randn_like` as we need the same size
        sample = mu + (eps * std) # sampling as if coming from the input space
        return sample

    def forward(self, x):
        # encoding
        x = F.relu(self.enc1(x))
        x = self.enc2(x).view(-1, 2, features)
        # get `mu` and `log_var`
        mu = x[:, 0, :] # the first feature values as mean
        log_var = x[:, 1, :] # the other feature values as variance
        # get the latent vector through reparameterization
        z = self.reparameterize(mu, log_var)

        # decoding
        x = F.relu(self.dec1(z))
        reconstruction = torch.sigmoid(self.dec2(x))
        return reconstruction, mu, log_var










# define a VAE
class VAE(nn.Module):
    def __init__(self):
        super(VAE, self).__init__()
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

        # encoder
        Enc_Layers = []
        Enc_Layers += [nn.Conv2d(in_channels=self.hist_len, out_channels=f_maps_1, kernel_size=f_size_1)]
        Enc_Layers += [nn.ReLU()]
        Enc_Layers += [nn.MaxPool2d(2,2)]
        Enc_Layers += [nn.Conv2d(in_channels=f_maps_1, out_channels=f_maps_2, kernel_size=f_size_2)]
        Enc_Layers += [nn.ReLU()]
        Enc_Layers += [nn.MaxPool2d(2,2)]

        self.enc1 = nn.Sequential(*Enc_Layers)

        Enc_Layers = []
        Enc_Layers += [nn.Linear(in_features=32*6*6, out_features=512)]
        Enc_Layers += [nn.ReLU()]
        Enc_Layers += [nn.Linear(in_features=512, out_features=features*2)]
        Enc_Layers += [nn.ReLU()]
        self.enc2 = nn.Sequential(*Enc_Layers)

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




        self.A = nn.Linear(in_features=features, out_features=features,bias=False)
        self.B = nn.Linear(in_features=1, out_features=features,bias=False)




    def reparameterize(self, mu, log_var):
        """
        :param mu: mean from the encoder's latent space
        :param log_var: log variance from the encoder's latent space
        """
        std = torch.exp(0.5*log_var) # standard deviation
        eps = torch.randn_like(std) # `randn_like` as we need the same size
        sample = mu + (eps * std) # sampling as if coming from the input space
        return sample

    def forward_single(self, x):
        # encoding
        x = torch.reshape(x,(x.size(0), self.hist_len, self.w, self.h))
        x = self.enc1(x)
        x = torch.reshape(x,(x.size(0),32*6*6))
        x = self.enc2(x).view(-1, 2, features)




        # get `mu` and `log_var`
        mu = x[:, 0, :] # the first feature values as mean
        log_var = x[:, 1, :] # the other feature values as variance
        # get the latent vector through reparameterization
        z = self.reparameterize(mu, log_var)




        # decoding
        reconstruction = self.dec1(z)
        reconstruction = torch.reshape(reconstruction,(reconstruction.size(0),32,9,9))
        reconstruction = self.dec2(reconstruction).view(-1, self.hist_len, self.w*self.h)

        return reconstruction, mu, log_var, z

    def forward(self, input):

        x = input[0]
        u = input[1]
        x_t1 = input[2]

        x_hat, mu, log_var, z = self.forward_single(x)

        x_hat_t1, mu_t1, log_var_t1, z_t1 = self.forward_single(x_t1)

        z_hat_t1 = self.A(z) + self.B(u)




        return x_hat, mu, log_var, z, x_hat_t1, mu_t1, log_var_t1, z_t1, z_hat_t1
