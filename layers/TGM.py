import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.distributions as dist

class tgm(nn.Module):
    def __init__(self, input_dim, latent_dim1, latent_dim2):
        super(tgm, self).__init__()

        self.fc_mu1 = nn.Linear(input_dim, latent_dim1)  
        self.fc_logvar1 = nn.Linear(input_dim, latent_dim1)  
        
        self.fc_mu2 = nn.Linear(latent_dim1, latent_dim2)  
        self.fc_logvar2 = nn.Linear(latent_dim1, latent_dim2)  
        
        self.fc1 = nn.Linear(latent_dim2, input_dim)
        self.fc2 = nn.Linear(input_dim, input_dim)

    def forward(self, x):
        
        mu1 = self.fc_mu1(x)
        logvar1 = self.fc_logvar1(x)
        std1 = torch.exp(0.5 * logvar1)
        eps1 = torch.randn_like(std1)
        z1 = mu1 + eps1 * std1
        
        mu2 = self.fc_mu2(z1)
        logvar2 = self.fc_logvar2(z1)
        std2 = torch.exp(0.5 * logvar2)
        eps2 = torch.randn_like(std2)
        z2 = mu2 + eps2 * std2
        
        x_recon = F.sigmoid(self.fc2(F.relu(self.fc1(z2))))
        return x_recon