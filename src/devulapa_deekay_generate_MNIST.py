"""
The preliminary work for phase-3 of the project includes image generation using Generative Adverserial Networks.
"""
# Authors: DeeKay Goswami & Naresh Kumar Devulapally

import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
from torchvision import datasets, transforms
from torchvision.utils import save_image
from tqdm import tqdm

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

batch_size = 64
latent_dim = 100
num_epochs = 50
image_size = 28
hidden_size = 256
learning_rate = 0.0002
beta1 = 0.5

class Generator(nn.Module):
    def __init__(self):
        super(Generator, self).__init__()
        self.main = nn.Sequential(
            nn.ConvTranspose2d(latent_dim, hidden_size * 4, 4, 1, 0, bias=False),
            nn.BatchNorm2d(hidden_size * 4),
            nn.ReLU(True),
            nn.ConvTranspose2d(hidden_size * 4, hidden_size * 2, 3, 2, 1, bias=False),
            nn.BatchNorm2d(hidden_size * 2),
            nn.ReLU(True),
            nn.ConvTranspose2d(hidden_size * 2, hidden_size, 4, 2, 1, bias=False),
            nn.BatchNorm2d(hidden_size),
            nn.ReLU(True),
            nn.ConvTranspose2d(hidden_size, 1, 4, 2, 1, bias=False),
            nn.Tanh()
        )

    def forward(self, x):
        return self.main(x)

class Discriminator(nn.Module):
    def __init__(self):
        super(Discriminator, self).__init__()
        self.main = nn.Sequential(
            nn.Conv2d(1, hidden_size, 4, 2, 1, bias=False),
            nn.LeakyReLU(0.2, inplace=True),
            nn.Conv2d(hidden_size, hidden_size * 2, 4, 2, 1, bias=False),
            nn.BatchNorm2d(hidden_size * 2),
            nn.LeakyReLU(0.2, inplace=True),
            nn.Conv2d(hidden_size * 2, hidden_size * 4, 3, 2, 1, bias=False),
            nn.BatchNorm2d(hidden_size * 4),
            nn.LeakyReLU(0.2, inplace=True),
            nn.Conv2d(hidden_size * 4, 1, 4, 1, 0, bias=False),
            nn.Sigmoid()
        )

    def forward(self, x):
        return self.main(x)

generator = Generator().to(device)
discriminator = Discriminator().to(device)

adversarial_loss = nn.BCELoss()

optimizer_G = optim.Adam(generator.parameters(), lr=learning_rate, betas=(beta1, 0.999))
optimizer_D = optim.Adam(discriminator.parameters(), lr=learning_rate, betas=(beta1, 0.999))

transform = transforms.Compose([
    transforms.ToTensor(),
    transforms.Normalize((0.5,), (0.5,))
])

mnist_dataset = datasets.MNIST(root="data", train=True, transform=transform, download=True)
dataloader = DataLoader(mnist_dataset, batch_size=batch_size, shuffle=True)
fixed_noise = torch.randn(100, latent_dim, 1, 1).to(device)

for epoch in range(num_epochs):
    for i, (images, _) in enumerate(tqdm(dataloader)):
        batch_size = images.size(0)
        images = images.to(device)

        discriminator.zero_grad()
        real_labels = torch.ones((batch_size, 1, 1, 1)).to(device)  
        real_output = discriminator(images)
        d_real_loss = adversarial_loss(real_output, real_labels)
        d_real_loss.backward()
        real_scores = real_output.mean().item()

        noise = torch.randn(batch_size, latent_dim, 1, 1).to(device)
        fake_images = generator(noise)
        fake_labels = torch.zeros((batch_size, 1, 1, 1)).to(device)
        fake_output = discriminator(fake_images.detach())
        d_fake_loss = adversarial_loss(fake_output, fake_labels)
        d_fake_loss.backward()
        fake_scores = fake_output.mean().item()

        d_loss = d_real_loss + d_fake_loss
        optimizer_D.step()

        generator.zero_grad()
        output = discriminator(fake_images)
        g_loss = adversarial_loss(output, real_labels)
        g_loss.backward()

        optimizer_G.step()

        if (i + 1) % 200 == 0:
            print(
                f"Epoch [{epoch+1}/{num_epochs}], Step [{i+1}/{len(dataloader)}], "
                f"D_loss: {d_loss.item():.4f}, G_loss: {g_loss.item():.4f}, "
                f"D(x): {real_scores:.2f}, D(G(z)): {fake_scores:.2f}"
            )

    if (epoch + 1) % 1 == 0:
        with torch.no_grad():
            fake_images = generator(fixed_noise).detach().cpu()
        save_image(fake_images, f"generated_images/MNIST/generated_images_epoch_{epoch+1}.png", nrow=10, normalize=True)
