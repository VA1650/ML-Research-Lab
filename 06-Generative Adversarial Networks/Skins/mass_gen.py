import torch
import torch.nn as nn
from torchvision.utils import save_image
import os

# Настройки как в обучении
latent_dim = 128
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

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

# Загружаем модель
gen = Generator().to(device)
gen.load_state_dict(torch.load("checkpoint_refined.pth", map_location=device))
gen.eval() # Переводим в режим проверки

# Создаем папку для лучших результатов
output_dir = "final_generated_skins"
os.makedirs(output_dir, exist_ok=True)

print("Генерирую 100 уникальных скинов...")

with torch.no_grad():
    for i in range(100):
        noise = torch.randn(1, latent_dim, 1, 1, device=device)
        fake_skin = gen(noise)
        # Сохраняем по одному
        save_image(fake_skin, f"{output_dir}/generated_{i:03d}.png", normalize=True)

print(f"Готово! Все скины лежат в папке: {output_dir}")