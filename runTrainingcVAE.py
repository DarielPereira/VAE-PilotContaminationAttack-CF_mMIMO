
import numpy as np
import torch
import random
from torch.utils.data import TensorDataset, DataLoader
from functionscVAE import VAEModel
from functionsUtils import drawingSetup


# ========================
# Training Configuration
# ========================
dataset_path = './TrainingData/cVAE_dataset_N_2_NbrSamples_1228330_ASD_10_P_100_Normalized_PsiInv_Large.npz'
device = 'cuda' if torch.cuda.is_available() else 'cpu'
batch_size = 32
n_epochs = 20
learning_rate = 1e-4
latent_dim = 6
hidden_dims = [16, 8]

# ========================
# Make training reproducible
# ========================
# Set a fixed seed
seed = 42
# Python random module
random.seed(seed)
# NumPy
np.random.seed(seed)
# PyTorch
torch.manual_seed(seed)
torch.cuda.manual_seed(seed)
torch.cuda.manual_seed_all(seed) # if using multi-GPU
torch.backends.cudnn.deterministic = True
torch.backends.cudnn.benchmark = False

# ========================
# Load Dataset
# ========================
data = np.load(dataset_path, allow_pickle=True)
B_real = data['B_samples']  # B normalizado y en representación real
R_real = data['R_samples']  # matriz de correlación en representación real

# Transform into numpy arrays
B_real = np.array(B_real)
R_real = np.array(R_real)

# Flatten
num_samples = B_real.shape[0]
input_dim = B_real.shape[1] * B_real.shape[2]
condition_dim = R_real.shape[1] * R_real.shape[2]

B_flat = B_real.reshape(num_samples, input_dim)
R_flat = R_real.reshape(num_samples, condition_dim)

# Transform into PyTorch tensors
B_tensor = torch.tensor(B_flat, dtype=torch.float32)
R_tensor = torch.tensor(R_flat, dtype=torch.float32)

# Create TensorDataset and DataLoader
dataset = TensorDataset(B_tensor, R_tensor)
dataloader = DataLoader(dataset, batch_size=batch_size, shuffle=True)

# ========================
# Instantiate VAE model
# ========================
vae = VAEModel(input_dim=input_dim, latent_dim=latent_dim, hidden_dims=hidden_dims)

# ========================
# Training
# ========================
print(f"Training on device: {device}")
vae.fit(dataloader, n_epochs=n_epochs, lr=learning_rate, device=device, verbose=True)

# ========================
# Save the trained model
# ========================
save_model_path = f'./Models/cVAE_model_NbrSamples_{num_samples}_ASD_10_P_100_Normalized_PsiInv_Large.pth'
vae.save_model(save_model_path)