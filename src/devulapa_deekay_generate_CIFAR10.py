# Authors: DeeKay Goswami & Naresh Kumar Devulapally

import torch
import torch.nn as nn
from tqdm import tqdm
import torch.optim as optim
import matplotlib.pyplot as plt
from torch.utils.data import DataLoader
from torchvision.utils import save_image
from torchvision import datasets, transforms

# Hyperparameters
batch_size = 64
latent_dim = 100
num_epochs = 100
image_size = 32
hidden_size = 64
learning_rate = 0.0002
beta1 = 0.5

transform = transforms.Compose([
    transforms.Resize(image_size),
    transforms.ToTensor(),
    transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))
])
cifar_dataset = datasets.CIFAR10(root="data", train=True, transform=transform, download=True)
dataloader = DataLoader(cifar_dataset, batch_size=batch_size, shuffle=True)
class_labels = cifar_dataset.classes

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

class Generator(nn.Module):
    def __init__(self):
        super(Generator, self).__init__()
        self.main = nn.Sequential(
            nn.ConvTranspose2d(latent_dim, hidden_size * 4, 4, 1, 0, bias=False),
            nn.BatchNorm2d(hidden_size * 4),
            nn.ReLU(True),
            nn.ConvTranspose2d(hidden_size * 4, hidden_size * 2, 4, 2, 1, bias=False),
            nn.BatchNorm2d(hidden_size * 2),
            nn.ReLU(True),
            nn.ConvTranspose2d(hidden_size * 2, hidden_size, 4, 2, 1, bias=False),
            nn.BatchNorm2d(hidden_size),
            nn.ReLU(True),
            nn.ConvTranspose2d(hidden_size, 3, 4, 2, 1, bias=False),
            nn.Tanh()
        )

    def forward(self, x):
        return self.main(x)

class Discriminator(nn.Module):
    def __init__(self):
        super(Discriminator, self).__init__()
        self.main = nn.Sequential(
            nn.Conv2d(3, hidden_size, 4, 2, 1, bias=False),
            nn.LeakyReLU(0.2, inplace=True),
            nn.Conv2d(hidden_size, hidden_size * 2, 4, 2, 1, bias=False),
            nn.BatchNorm2d(hidden_size * 2),
            nn.LeakyReLU(0.2, inplace=True),
            nn.Conv2d(hidden_size * 2, hidden_size * 4, 4, 2, 1, bias=False),
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

for epoch in range(num_epochs):
    for i, (images, labels) in enumerate(tqdm(dataloader)):
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
            fixed_noise = torch.randn(100, latent_dim, 1, 1).to(device)
            fake_images = generator(fixed_noise).detach().cpu()

        fig, axs = plt.subplots(10, 10, figsize=(10, 10))
        fig.suptitle(f"Generated Images - Epoch {epoch+1}", fontsize=16)

        for i in range(10):
            for j in range(10):
                image = fake_images[i * 10 + j]
                image = image * 0.5 + 0.5
                image = image.permute(1, 2, 0)
                axs[i, j].imshow(image)
                axs[i, j].axis("off")
                axs[i, j].set_title(class_labels[j])

        plt.tight_layout()
        plt.savefig(f"generated_images/CIFAR10/generated_images_CIFAR_epoch_{epoch+1}.png")
        plt.close(fig)