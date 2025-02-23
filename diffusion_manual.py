import torch
import torch.nn as nn
import torch.nn.functional as F
from torchvision import datasets as dset
import torchvision.transforms as transforms
import numpy as np
from torch.utils.data import DataLoader
import matplotlib.pyplot as plt
import torchvision.transforms.functional as TF

class Noise_schedular():
    def __init__(self, num_steps=100, beta_start=1e-4, beta_end=1e-3):
        self.beta = torch.linspace(beta_start, beta_end, num_steps)
        self.alpha = 1 - self.beta
        self.cumprod_alpha = torch.cumprod(self.alpha, dim=0)
        self.sqrt_cumprod_alpha = torch.sqrt(self.cumprod_alpha)
        self.sqrt_alpha = torch.sqrt(self.alpha)
    def add_noise(self, x0, t):
        noise = torch.randn_like(x0)
        scaled_x = self.sqrt_cumprod_alpha[t - 1] * x0
        x_with_noise = scaled_x + torch.sqrt(1 - self.sqrt_cumprod_alpha[t - 1]) * noise
        return x_with_noise, noise
    
def flatten(x):
    N = x.shape[0]
    return x.view(N, -1)

def flat_reverse(x):
    N = x.shape[0]
    return x.view(N, 32, 32, 32)

class Unet(nn.Module):
    def __init__(self):
        super().__init__()
        self.Conv2d1 = nn.Conv2d(3, 32, 3, 1, 1)
        self.Conv2d2 = nn.Conv2d(32, 32, 3, 1, 1)
        self.Linear = nn.Linear(32 * 32 * 32, 32 * 32 * 32)
        self.ReLU = nn.ReLU()
        self.tConv2d1 = nn.ConvTranspose2d(32, 32, 3, 1, padding=1)
        self.tConv2d2 = nn.ConvTranspose2d(32, 3, 3, 1, padding=1)
        self.time_encoding = nn.Linear(1, 32*32*32)
    def forward(self, x, t):
        t_encoded = self.time_encoding(t.unsqueeze(-1).float())
        x = self.Conv2d1(x)
        x = self.Conv2d2(x)
        x = flatten(x)
        x = self.Linear(x)
        x = x * t_encoded
        x = self.ReLU(x)
        x = flat_reverse(x)
        x = self.tConv2d1(x)
        x = self.tConv2d2(x)
        return x
def train(Noise_schedular, dataLoader, optimizer, device, model, epochs=10):
    model.train()
    for epoch in range(epochs):
        print(f"Epoch {epoch + 1}")
        for i, (x, _) in enumerate(dataLoader):
            x = x.to(device)
            t = torch.randint(0, 100, (1,)).to(device)
            x_with_noise, noise = Noise_schedular.add_noise(x, t)
            optimizer.zero_grad()
            output = model(x_with_noise, t)
            loss = F.mse_loss(output, noise)
            loss.backward()
            optimizer.step()
            if i % 100 == 0:
                print(f"Iteration: {i}")
                print(f"Loss: {loss.item()}")
                print()
@torch.no_grad()
def sample(model, Noise_schedular, device, n_samples=1):
    model.eval()
    x = torch.randn(n_samples, 3, 32, 32).to(device)
    for t in reversed(range(100)):
        t_batch = torch.tensor([t], device=device).repeat(n_samples)
        predicted_noise = model(x, t_batch)
        alpha = Noise_schedular.alpha[t]
        alpha_bar = Noise_schedular.cumprod_alpha[t]
        beta = Noise_schedular.beta[t]
        if t > 0:
            noise = torch.randn_like(x)
        else:
            noise = 0
        x = 1 / torch.sqrt(alpha) * (x - ((1 - alpha) / torch.sqrt(1 - alpha_bar)) * predicted_noise) + torch.sqrt(beta) * noise
    return x
def main():
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f'Using device:{device}')
    transform = transforms.Compose([transforms.Resize((32, 32)), transforms.ToTensor()])
    dataset = dset.CIFAR10(root="./data", transform=transform, download=True)
    dataLoader = DataLoader(dataset, batch_size=1, shuffle=True)
    Noise_schedulars = Noise_schedular()
    model = Unet().to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr=1e-3)
    train(Noise_schedulars, dataLoader, optimizer, device, model)
    sample_image = sample(model, Noise_schedulars, device)
    print(sample_image.shape)
    img = sample_image[0].cpu()

# 将张量转换为 PIL 图像
    img_pil = TF.to_pil_image(img)

# 显示图像
    plt.imshow(img_pil)
    plt.axis('off')
    plt.show()
if __name__ == "__main__":
    main()
        
        
