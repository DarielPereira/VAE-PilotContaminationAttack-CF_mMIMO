# cVAE.py
import torch
import torch.nn as nn
import torch.optim as optim
import os


class CVAEModel(nn.Module):
    """
    Conditional VAE for channel estimation anomaly detection.
    Input: B (real-valued covariance of estimate)
    Condition: R (real-valued channel correlation matrix)
    """

    def __init__(self, input_dim, condition_dim, latent_dim=8, hidden_dims=[128, 64]):
        """
        :param input_dim: dimension of flattened B (real-valued)
        :param condition_dim: dimension of flattened R (real-valued)
        :param latent_dim: size of latent vector z
        :param hidden_dims: list of hidden layer sizes
        """
        super(CVAEModel, self).__init__()

        self.input_dim = input_dim
        self.condition_dim = condition_dim
        self.latent_dim = latent_dim

        # Encoder: q(z|B,R)
        encoder_layers = []
        prev_dim = input_dim + condition_dim
        for h_dim in hidden_dims:
            encoder_layers.append(nn.Linear(prev_dim, h_dim))
            encoder_layers.append(nn.ReLU())
            prev_dim = h_dim
        self.encoder = nn.Sequential(*encoder_layers)

        # Latent space parameters
        self.fc_mu = nn.Linear(prev_dim, latent_dim)
        self.fc_logvar = nn.Linear(prev_dim, latent_dim)

        # Decoder: p(B|z,R)
        decoder_layers = []
        prev_dim = latent_dim + condition_dim
        for h_dim in reversed(hidden_dims):
            decoder_layers.append(nn.Linear(prev_dim, h_dim))
            decoder_layers.append(nn.ReLU())
            prev_dim = h_dim
        decoder_layers.append(nn.Linear(prev_dim, input_dim))
        self.decoder = nn.Sequential(*decoder_layers)

    def encode(self, x, cond):
        """
        Encode input x with condition cond to latent parameters
        """
        h = torch.cat([x, cond], dim=1)
        h = self.encoder(h)
        mu = self.fc_mu(h)
        logvar = self.fc_logvar(h)
        return mu, logvar

    def reparameterize(self, mu, logvar):
        """
        Reparameterization trick: sample z ~ N(mu, sigma^2)
        """
        std = torch.exp(0.5 * logvar)
        eps = torch.randn_like(std)
        return mu + eps * std

    def decode(self, z, cond):
        """
        Decode latent z with condition cond to reconstruct B
        """
        h = torch.cat([z, cond], dim=1)
        x_recon = self.decoder(h)
        return x_recon

    def forward(self, x, cond):
        mu, logvar = self.encode(x, cond)
        z = self.reparameterize(mu, logvar)
        x_recon = self.decode(z, cond)
        return x_recon, mu, logvar

    @staticmethod
    def loss_function(recon_x, x, mu, logvar):
        """
        Compute VAE loss = reconstruction + KL divergence
        """
        recon_loss = nn.MSELoss(reduction='sum')(recon_x, x)
        # KL divergence
        kld = -0.5 * torch.sum(1 + logvar - mu.pow(2) - logvar.exp())
        return recon_loss + kld, recon_loss, kld

    def fit(self, dataloader, n_epochs=100, lr=1e-3, device='cpu',
            beta_max=1.0, kl_anneal_epochs=50, verbose=True):
        """
        Train the cVAE with beta-VAE and KL annealing.
        beta_max: maximum weight for KL term
        kl_anneal_epochs: number of epochs to linearly increase KL weight
        """
        self.to(device)
        optimizer = optim.Adam(self.parameters(), lr=lr)
        self.train()

        for epoch in range(1, n_epochs + 1):
            total_loss = 0
            recon_total = 0
            kl_total = 0

            # Linear KL annealing
            # beta = beta_max * min(1.0, epoch / kl_anneal_epochs)
            beta = 0.2

            for x_batch, cond_batch in dataloader:
                x_batch, cond_batch = x_batch.to(device), cond_batch.to(device)
                optimizer.zero_grad()
                x_recon, mu, logvar = self.forward(x_batch, cond_batch)
                recon_loss = nn.MSELoss(reduction='sum')(x_recon, x_batch)
                kl = -0.5 * torch.sum(1 + logvar - mu.pow(2) - logvar.exp())
                loss = recon_loss + beta * kl
                loss.backward()
                optimizer.step()

                total_loss += loss.item()
                recon_total += recon_loss.item()
                kl_total += kl.item()

            if verbose and epoch % 10 == 0:
                print(f"Epoch [{epoch}/{n_epochs}] "
                      f"Loss: {total_loss:.2f}, Recon: {recon_total:.2f}, KL: {kl_total:.2f}, Beta: {beta:.2f}")

    def reconstruct(self, x, cond, device='cpu'):
        """
        Reconstruct B given x and condition R
        """
        self.to(device)
        self.eval()
        with torch.no_grad():
            x = x.to(device)
            cond = cond.to(device)
            x_recon, _, _ = self.forward(x, cond)
        return x_recon.cpu()

    def save_model(self, path):
        os.makedirs(os.path.dirname(path), exist_ok=True)
        torch.save(self.state_dict(), path)
        print(f"Model saved to {path}")

    def load_model(self, path, map_location='cpu'):
        self.load_state_dict(torch.load(path, map_location=map_location))
        print(f"Model loaded from {path}")