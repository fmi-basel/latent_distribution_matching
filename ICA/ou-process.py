import torch
import torch.nn as nn
import torch.optim as optim
import numpy as np
from matplotlib.image import imread
import matplotlib.pyplot as plt
plt.rcParams.update({'font.size': 6})

# seed everything
torch.manual_seed(1)
np.random.seed(1)

def rotation_matrix_n(params, n):
    # Initialize skew-symmetric matrix
    A = torch.zeros(n, n, device=params.device, dtype=params.dtype)
    
    # Fill upper triangular part with parameters
    idx = 0
    for i in range(n):
        for j in range(i + 1, n):
            A[i, j] = params[..., idx]
            A[j, i] = -params[..., idx]  # Ensure skew-symmetry
            idx += 1
    
    # Compute matrix exponential
    R = torch.matrix_exp(A)
    return R

def center(x):
    mean = torch.mean(x, dim=1, keepdim=True)
    centered = x - mean
    return centered, mean

def covariance(x):
    n = x.shape[1] - 1
    mean = torch.mean(x, dim=1, keepdim=True)
    m = x - mean
    return (m @ m.T) / n


def plot_hist(images, title):
    im1 = images[0]
    im2 = images[1]
    fig = plt.figure()
    plt.scatter(im1, im2, s=1)
    plt.savefig(f'out/{title}.png')
    fig = plt.figure()
    plt.hist(im1, bins=256, alpha=0.5, label='Image 1', color='r')
    plt.hist(im2, bins=256, alpha=0.5, label='Image 2', color='b')
    plt.savefig(f'out/{title}_hist.png')






class ICA(nn.Module):
    # 1. Define the Unmixing Matrix as a learnable parameter
    def __init__(self, num_pixels, num_components, vars):
        super(ICA, self).__init__()
        # parameters of the rotation matrix
        self.params = nn.Parameter(torch.rand(num_components * (num_components - 1) // 2) * 0.01)
        self.n = num_components
        self.vars = vars

    def forward(self, x):
        # The forward pass simply unmixes the data
        # make sure that W is a rotation
        W = rotation_matrix_n(self.params, self.n)
        return torch.mm(W, x)

    # 2. Define the distribution matching loss function
    def energy(self, Ss):
        g_S = 0
        for i in range(self.n):
            g_S -= torch.pow(Ss[i], 2) / torch.sqrt(self.vars[i]) # Gaussian with variable variance
        W = rotation_matrix_n(self.params, self.n)
        log_W = torch.logdet(W @ W.T + 1e-8) 
        return 0.5 * log_W + torch.mean(g_S)

def train_ica(mixed_data, num_components, vars):
    # 3. Training Loop
    model = ICA(num_pixels=mixed_data.shape[1], num_components=num_components, vars=vars)
    optimizer = optim.SGD(model.parameters(), lr=0.01)

    for epoch in range(1000):
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
            W = rotation_matrix_n(model.params, model.n)
            print(f"W: {W.detach().numpy()}")
    W = rotation_matrix_n(model.params, model.n)
    return W, separated_sources



# Parameters for the Ornstein-Uhlenbeck process
theta = 2.0  # Mean reversion rate
mu = 0.0     # Long-term mean
sigma = 0.3  # Volatility
x0 = 0.0     # Initial value
T = 5000.0     # Total time
dt = 0.1    # Time step
N = int(T / dt)  # Number of time steps

# Time array
t = np.linspace(0, T, N)

# Simulate Ornstein-Uhlenbeck paths
def simulate_ou(theta, mu, sigma, x0, N, dt):
    paths = np.zeros((N))
    paths[0] = x0  # Set initial value for all paths
    
    # Generate random increments (Gaussian noise)
    dW = np.random.normal(0, np.sqrt(dt), size=(N-1))
    
    # Euler-Maruyama method for OU process
    for i in range(1, N):
        paths[i] = paths[i-1] + theta * (mu - paths[i-1]) * dt + sigma * dW[i-1]
    
    return paths

def gaussian_pdf(x, mean=0.0, std=1.0):
    coeff = 1.0 / (std * np.sqrt(2 * np.pi))
    exponent = -0.5 * ((x - mean) / std) ** 2
    return coeff * np.exp(exponent)

# Simulate paths
ou_path = simulate_ou(theta, mu, sigma, x0, N, dt)
ou_path_slow = simulate_ou(0.1*theta, mu, sigma, x0, N, dt)

# we will need this to define the right latent model
vars = torch.tensor([np.var(ou_path), np.var(ou_path_slow)]).float()

S = torch.stack([torch.tensor(ou_path), torch.tensor(ou_path_slow)], dim=0).float()  # shape (2, N)
S_centered, mean = center(S)

# Random mixing matrix with rotation matrix
w = np.pi / 4  # 45 degrees
A = torch.tensor([[np.cos(w), -np.sin(w)], [np.sin(w), np.cos(w)]]).float()

# Mix the sources
X_mixed = A @ S

# Perform ICA unmixing
X_centered, mean = center(X_mixed)
# Train the ICA model
W, separated_sources = train_ica(X_centered, num_components=2, vars=vars)
# Recover sources (up to scaling and permutation)
S_recovered = W @ X_centered
recovered_path1 = S_recovered[0].detach().numpy()
recovered_path2 = S_recovered[1].detach().numpy()

# compare
orig = np.concatenate([ou_path.flatten(), ou_path_slow.flatten()])  # shape (num_pixels)
rec = np.concatenate([recovered_path1.flatten(), recovered_path2.flatten()])  # shape (num_pixels)
mix = np.concatenate([X_centered[0].numpy().flatten(), X_centered[1].numpy().flatten()])  # shape (num_pixels)

fig = plt.figure()
plt.hist(orig, bins=50, alpha=0.5, label='Original image', color='g')
plt.hist(mix, bins=50, alpha=0.5, label='Mixed image', color='r')
plt.hist(rec, bins=50, alpha=0.5, label='Reconstructed image', color='b')
plt.legend()
plt.savefig(f'out/ou_compare_hist.png')

def nice_axis(ax, bottom=True):
    ax.spines['top'].set_visible(False)
    ax.spines['right'].set_visible(False)
    if not bottom:
        ax.spines['bottom'].set_visible(False)
        ax.tick_params(labelbottom=False, bottom=False)

# Plot the paths
t_max = 1000  # Plot only the first 1000 time steps for clarity
fig, axs = plt.subplots(3, 2, figsize=(2, 2.5), gridspec_kw={'width_ratios': [6, 1]})
axs[0,0].plot(t[:t_max], ou_path[:t_max], label='Original OU Path (Fast)', color='g')
axs[0,0].plot(t[:t_max], ou_path_slow[:t_max], label='Original OU Path (Slow)', color='b')
# axs[0,0].set_title('Original OU Paths')
axs[0,0].set_ylim([-2, 2])
axs[0,0].set_ylabel('OU proc.')
nice_axis(axs[0,0], bottom=False)
axs[0,1].hist(ou_path, bins=40, orientation='horizontal', color='g', alpha=0.5, density=True)
axs[0,1].hist(ou_path_slow, bins=40, orientation='horizontal', color='b', alpha=0.5, density=True)
axs[0,1].axis('off')
axs[0,1].set_ylim([-2, 2])

axs[1,0].sharex(axs[0,0])
axs[1,0].plot(t[:t_max], X_mixed[0].numpy()[:t_max], label='Mixed Path 1', color='r')
axs[1,0].plot(t[:t_max], X_mixed[1].numpy()[:t_max], label='Mixed Path 2', color='orange')
# axs[1,0].set_title('Mixed Paths')
axs[1,0].set_ylim([-2, 2])
axs[1,0].set_ylabel('mixed proc.')
nice_axis(axs[1,0], bottom=False)
axs[1,1].hist(X_mixed[0].numpy(), bins=40, orientation='horizontal', color='r', alpha=0.5, density=True)
axs[1,1].hist(X_mixed[1].numpy(), bins=40, orientation='horizontal', color='orange', alpha=0.5, density=True)
axs[1,1].axis('off')
axs[1,1].set_ylim([-2, 2])

axs[2,0].sharex(axs[0,0])
axs[2,0].plot(t[:t_max], recovered_path1[:t_max], label='Recovered Path 1', color='olive')
axs[2,0].plot(t[:t_max], recovered_path2[:t_max], label='Recovered Path 2', color='purple')
# axs[2,0].set_title('Recovered OU Paths after ICA')
axs[2,0].set_ylim([-2, 2])
axs[2,0].set_ylabel('recovered proc.')
axs[2,0].set_xlabel('Time')
nice_axis(axs[2,0])
axs[2,1].hist(recovered_path1, bins=40, orientation='horizontal', color='olive', alpha=0.5, density=True)
axs[2,1].hist(recovered_path2, bins=40, orientation='horizontal', color='purple', alpha=0.5, density=True)
x = np.linspace(-3, 3, 1000)
pdf = gaussian_pdf(x, mean=0.0, std=np.sqrt(vars[0].item()))
pdf2 = gaussian_pdf(x, mean=0.0, std=np.sqrt(vars[1].item()))
axs[2,1].plot(pdf, x, '--', color='red', label='Theoretical PDF', linewidth=0.7)  # plot the pdf using
axs[2,1].plot(pdf2, x, '--', color='red', linewidth=0.7)  # plot the pdf using
axs[2,1].axis('off')
axs[2,1].set_ylim([-2, 2])

plt.xlabel('Time')
plt.tight_layout()
plt.savefig(f'out/ou_paths.pdf')
