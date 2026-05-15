import torch
import torch.nn as nn
import torch.optim as optim
from torchvision import transforms, datasets
from torch.utils.data import DataLoader
from torchvision.utils import save_image
import matplotlib.pyplot as plt
import os
from PIL import Image

# --- НАСТРОЙКИ ---
latent_dim = 128
batch_size = 64
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
data_path = r"D:\bckp2\Программы\Мои\скины\skins" # Путь к папке, где лежит папка со скинами

# --- МОДЕЛИ (Твоя архитектура DCGAN) ---
class Generator(nn.Module):
    def __init__(self):
        super(Generator, self).__init__()
        self.main = nn.Sequential(
            nn.ConvTranspose2d(latent_dim, 512, 4, 1, 0, bias=False),
            nn.BatchNorm2d(512), nn.ReLU(True),
            nn.ConvTranspose2d(512, 256, 4, 2, 1, bias=False),
            nn.BatchNorm2d(256), nn.ReLU(True),
            nn.ConvTranspose2d(256, 128, 4, 2, 1, bias=False),
            nn.BatchNorm2d(128), nn.ReLU(True),
            nn.ConvTranspose2d(128, 64, 4, 2, 1, bias=False),
            nn.BatchNorm2d(64), nn.ReLU(True),
            nn.ConvTranspose2d(64, 4, 4, 2, 1, bias=False),
            nn.Tanh()
        )
    def forward(self, x): return self.main(x)

class Discriminator(nn.Module):
    def __init__(self):
        super(Discriminator, self).__init__()
        self.main = nn.Sequential(
            nn.Conv2d(4, 64, 4, 2, 1, bias=False), nn.LeakyReLU(0.2, inplace=True),
            nn.Conv2d(64, 128, 4, 2, 1, bias=False), nn.BatchNorm2d(128), nn.LeakyReLU(0.2, inplace=True),
            nn.Conv2d(128, 256, 4, 2, 1, bias=False), nn.BatchNorm2d(256), nn.LeakyReLU(0.2, inplace=True),
            nn.Conv2d(256, 512, 4, 2, 1, bias=False), nn.BatchNorm2d(512), nn.LeakyReLU(0.2, inplace=True),
            nn.Conv2d(512, 1, 4, 1, 0, bias=False), nn.Sigmoid()
        )
    def forward(self, x): return self.main(x).view(-1)

# --- ПОДГОТОВКА ---
def pad_to_64(img):
    if img.size == (64, 32):
        new_img = Image.new("RGBA", (64, 64), (0, 0, 0, 0))
        new_img.paste(img, (0, 0))
        return new_img
    return img.resize((64, 64), Image.NEAREST) if img.size != (64, 64) else img

transform = transforms.Compose([
    transforms.Lambda(lambda img: img.convert("RGBA")),
    transforms.Lambda(pad_to_64),
    transforms.ToTensor(),
    transforms.Normalize((0.5, 0.5, 0.5, 0.5), (0.5, 0.5, 0.5, 0.5)),
])

dataset = datasets.ImageFolder(root=data_path, transform=transform)
dataloader = DataLoader(dataset, batch_size=batch_size, shuffle=True)

gen = Generator().to(device)
disc = Discriminator().to(device)

# ЗАГРУЗКА ВЕСОВ ИЗ ЧИСТОЙ 200 ЭПОХИ
if os.path.exists("checkpoint_refined.pth"):
    gen.load_state_dict(torch.load("checkpoint_refined.pth", map_location=device))
    print("Загружены базовые веса (240 эпох). Начинаем мягкую доводку...")
else:
    print("Ошибка: файл checkpoint_refined.pth не найден!")
    exit()

# РАЗНЫЕ СКОРОСТИ ОБУЧЕНИЯ
optimizer_G = optim.Adam(gen.parameters(), lr=0.00002, betas=(0.5, 0.999))
optimizer_D = optim.Adam(disc.parameters(), lr=0.000005, betas=(0.5, 0.999))
criterion = nn.BCELoss()

G_losses, D_losses = [], []

# --- ОБУЧЕНИЕ ---
for epoch in range(240, 301):
    for i, (real_imgs, _) in enumerate(dataloader):
        b_size = real_imgs.size(0)
        real_imgs = real_imgs.to(device)
        
        # Дискриминатор: Label Smoothing (0.9 вместо 1.0)
        disc.zero_grad()
        label = torch.full((b_size,), 0.9, device=device) 
        output = disc(real_imgs)
        errD_real = criterion(output, label)
        errD_real.backward()

        noise = torch.randn(b_size, latent_dim, 1, 1, device=device)
        fake = gen(noise)
        label.fill_(0.0)
        output = disc(fake.detach())
        errD_fake = criterion(output, label)
        errD_fake.backward()
        optimizer_D.step()

        # Генератор
        gen.zero_grad()
        label.fill_(1.0) # Для генератора цель всё еще идеальный обман
        output = disc(fake)
        errG = criterion(output, label)
        errG.backward()
        optimizer_G.step()

        G_losses.append(errG.item())
        D_losses.append(errD_real.item() + errD_fake.item())

    print(f"[{epoch}/300] D_Loss: {D_losses[-1]:.4f} G_Loss: {errG.item():.4f}")

    if epoch % 10 == 0:
        plt.figure(figsize=(10,5)); plt.plot(G_losses, label="G"); plt.plot(D_losses, label="D")
        plt.legend(); plt.savefig("refined_plot.png"); plt.close()
        save_image(fake.data[:16], f"refined_sample_{epoch}.png", nrow=4, normalize=True)
        torch.save(gen.state_dict(), "checkpoint_refined.pth")

print("Готово! Проверь 'refined_sample_210.png' — результат должен быть чище.")