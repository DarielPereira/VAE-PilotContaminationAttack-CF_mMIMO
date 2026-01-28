# cVAE.py
import torch
import torch.nn as nn
import torch.optim as optim
import os


import torch
import torch.nn as nn
import torch.optim as optim
import os

class VAEModel(nn.Module):
    """
    Unconditioned VAE for small covariance matrices (2x2).
    Input: flattened covariance matrix (real-valued)
    """

    def __init__(self, input_dim=4, latent_dim=6, hidden_dims=[16, 8], beta=0.2, lambda_fb=0.2):
        super(VAEModel, self).__init__()

        self.input_dim = input_dim
        self.latent_dim = latent_dim
        self.hidden_dims = hidden_dims
        self.beta = beta
        self.lambda_fb = lambda_fb

        # Encoder
        encoder_layers = []
        prev_dim = input_dim
        for h_dim in hidden_dims:
            encoder_layers.append(nn.Linear(prev_dim, h_dim))
            encoder_layers.append(nn.ReLU())
            prev_dim = h_dim
        self.encoder = nn.Sequential(*encoder_layers)

        # Latent parameters
        self.fc_mu = nn.Linear(prev_dim, latent_dim)
        self.fc_logvar = nn.Linear(prev_dim, latent_dim)

        # Decoder
        decoder_layers = []
        prev_dim = latent_dim
        for h_dim in reversed(hidden_dims):
            decoder_layers.append(nn.Linear(prev_dim, h_dim))
            decoder_layers.append(nn.ReLU())
            prev_dim = h_dim
        decoder_layers.append(nn.Linear(prev_dim, input_dim))
        self.decoder = nn.Sequential(*decoder_layers)

    def encode(self, x):
        h = self.encoder(x)
        mu = self.fc_mu(h)
        logvar = self.fc_logvar(h)
        return mu, logvar

    def reparameterize(self, mu, logvar):
        std = torch.exp(0.5 * logvar)
        eps = torch.randn_like(std)
        return mu + eps * std

    def decode(self, z):
        return self.decoder(z)

    def forward(self, x):
        mu, logvar = self.encode(x)
        z = self.reparameterize(mu, logvar)
        x_recon = self.decode(z)
        return x_recon, mu, logvar

    def compute_loss(self, x, reduction='mean'):
        x_recon, mu, logvar = self.forward(x)
        # Reconstruction
        recon_loss = nn.MSELoss(reduction='sum')(x_recon, x) / x.size(0)
        # KL per dimension
        kl_per_dim = -0.5 * (1 + logvar - mu.pow(2) - logvar.exp())
        kl_free = torch.clamp(kl_per_dim, min=self.lambda_fb)
        kl_loss = kl_free.sum(dim=1)
        if reduction == 'mean':
            kl_loss = kl_loss.mean()
        loss = recon_loss + self.beta * kl_loss
        return loss, recon_loss, kl_loss

    def fit(self, dataloader, n_epochs=200, lr=1e-3, device='cpu', verbose=True):
        self.to(device)
        optimizer = optim.Adam(self.parameters(), lr=lr)
        self.train()

        for epoch in range(1, n_epochs+1):
            total_loss = 0
            recon_total = 0
            kl_total = 0

            for x_batch,_ in dataloader:
                x_batch = x_batch.to(device)
                optimizer.zero_grad()
                loss, recon_loss, kl_loss = self.compute_loss(x_batch)
                loss.backward()
                optimizer.step()

                total_loss += loss.item()
                recon_total += recon_loss.item()/x_batch.size(0)
                kl_total += kl_loss.item()/x_batch.size(0)

            # if verbose and epoch % 10 == 0:
            print(f"Epoch [{epoch}/{n_epochs}] "
                  f"Loss: {total_loss:.2f}, Recon: {recon_total:.2f}, KL: {kl_total:.2f}, "
                  f"Beta: {self.beta:.2f}, Lambda_fb: {self.lambda_fb:.2f}")

    def reconstruct(self, x, device='cpu'):
        self.to(device)
        self.eval()
        with torch.no_grad():
            x = x.to(device)
            x_recon, _, _ = self.forward(x)
        return x_recon.cpu()

    def save_model(self, path):
        os.makedirs(os.path.dirname(path), exist_ok=True)
        torch.save(self.state_dict(), path)
        print(f"Model saved to {path}")

    def load_model(self, path, map_location='cpu'):
        self.load_state_dict(torch.load(path, map_location=map_location))
        print(f"Model loaded from {path}")