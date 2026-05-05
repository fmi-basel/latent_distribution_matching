import torch
import torch.nn as nn
import torch.optim as optim
import numpy as np
from matplotlib.image import imread
import matplotlib.pyplot as plt
import scipy
# fontsize for plots
plt.rcParams.update({'font.size': 6})


def center(x):
    mean = torch.mean(x, dim=1, keepdim=True)
    centered = x - mean
    return centered, mean

def covariance(x):
    n = x.shape[1] - 1
    mean = torch.mean(x, dim=1, keepdim=True)
    m = x - mean
    return (m @ m.T) / n


class ICA(nn.Module):
    # 1. Define the Unmixing Matrix as a learnable parameter
    def __init__(self, num_pixels, num_components):
        super(ICA, self).__init__()
        # W is the unmixing matrix, initialized randomly
        self.W = nn.Parameter(torch.eye(num_components, num_components))

    def forward(self, x):
        # The forward pass simply unmixes the data
        return torch.mm(self.W, x)

    # 2. Define the distribution matching loss function
    def energy(self, S):
        g_S = - torch.pow(torch.abs(S), 4) * 20 # generalized Gaussian with beta=4
        log_W = torch.logdet(self.W @ self.W.T + 1e-8)   # Add epsilon for stability
        return 0.5 * log_W + torch.sum(torch.mean(g_S, dim=1))

def generalized_gaussian_pdf(x, beta=4, scale=1.0):
    coeff = beta / (2 * scale * scipy.special.gamma(1 / beta))
    exponent = - (np.abs(x) / scale) ** beta
    return coeff * np.exp(exponent)

def train_ica(mixed_data, num_components):
    # 3. Training Loop
    model = ICA(num_pixels=mixed_data.shape[1], num_components=num_components)
    optimizer = optim.SGD(model.parameters(), lr=0.003)

    for epoch in range(20000):
        # Zero gradients
        optimizer.zero_grad()

        # Forward pass: get separated components
        separated_sources = model(mixed_data)

        # Calculate loss
        loss = - model.energy(separated_sources)

        # Backward pass: compute gradients
        loss.backward()

        # Update parameters
        optimizer.step()

        # Print loss for monitoring
        if epoch % 100 == 0:
            print(f"Epoch {epoch}, Loss: {loss.item()}")
            print(f"W: {model.W.detach().numpy()}")
    return model.W, separated_sources



# import images and preprocess them into a 2D tensor
# Assume images are flattened and stacked into a tensor of shape (num_pixels, num_samples)
height = 200
width = 200

im1 = imread('data/1.png').flatten()
print(im1.shape)
im2 = imread('data/2.png').flatten()
im1 = torch.tensor(im1)
im2 = torch.tensor(im2)

S = torch.stack([im1, im2], dim=0).float()  # shape (2, num_pixels)

S_centered, mean = center(S)
original_img1 = S_centered[0].reshape(height, width, 3).numpy()
original_img2 = S_centered[1].reshape(height, width, 3).numpy()

# Random mixing matrix
A = torch.tensor([[1.0, 0.7], [0.7, 1.0]])

# Mix the sources
X_mixed = A @ S

# Reshape back to images for visualization
mixed_img1 = X_mixed[0].reshape(height, width, 3).numpy()
mixed_img2 = X_mixed[1].reshape(height, width, 3).numpy()

# Perform ICA unmixing
X_centered, mean = center(X_mixed)
# Train the ICA model
W, separated_sources = train_ica(X_centered, num_components=2)
# Recover sources (up to scaling and permutation)
S_recovered = W @ X_centered
# Reshape back to images for visualization
recovered_img1 = S_recovered[0].reshape(height, width, 3).detach().numpy()
recovered_img2 = S_recovered[1].reshape(height, width, 3).detach().numpy()

# compare
orig = np.concatenate([original_img1.flatten(), original_img2.flatten()])  # shape (num_pixels)
rec = np.concatenate([recovered_img1.flatten(), recovered_img2.flatten()])  # shape (num_pixels)
mix = np.concatenate([X_centered[0].numpy().flatten(), X_centered[1].numpy().flatten()])  # shape (num_pixels)

fig, ax = plt.subplots(1, 3, figsize=(2.8, 0.9))
# share y axis
for a in ax:
    a.sharey(ax[0])
    a.spines['top'].set_visible(False)
    a.spines['right'].set_visible(False)
    a.set_xlabel('Pixel intensity')
ax[0].set_ylabel('Probability')
orig = orig / np.std(orig)
rec = rec / np.std(rec)
mix = mix / np.std(mix)
ax[0].hist(orig, bins=50, alpha=0.3, label='Original image', color='g', density=True)
ax[1].hist(mix, bins=50, alpha=0.3, label='Mixed image', color='r', density=True)
ax[2].hist(rec, bins=50, alpha=0.3, label='Reconstructed image', color='b', density=True)
x = np.linspace(-3, 3, 1000)
pdf = generalized_gaussian_pdf(torch.tensor(x), beta=4, scale=1.0).numpy()
plt.plot(x, pdf, '--', color='red', label='Generalized Gaussian (β=4)', linewidth=0.7)  # plot the pdf using

# plt.legend(frameon=False)
plt.tight_layout()
plt.savefig(f'out/compare_hist.pdf')

# Save the images
def rescale_01(arr):
    return (arr - arr.min()) / (arr.max() - arr.min())
def clip_and_save(image, filename):
    plt.imsave("out/img_{}.png".format(filename), np.clip(rescale_01(image), 0, 1))
clip_and_save(original_img1, 'original1')
clip_and_save(original_img2, 'original2')
clip_and_save(mixed_img1, 'mixed1')
clip_and_save(mixed_img2, 'mixed2')
clip_and_save(recovered_img1, 'recovered1')
clip_and_save(recovered_img2, 'recovered2')
