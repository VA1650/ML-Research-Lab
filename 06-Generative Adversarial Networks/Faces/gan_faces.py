
import os
from PIL import Image
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader
import torchvision.transforms as transforms
import torchvision.utils as vutils
import matplotlib.pyplot as plt
from tqdm import tqdm

# --- Параметры ---
DATA_DIR = 'D:/bckp2/Университет/ИИ Руденко 4 курс/5/faces/'  # Путь к директории с изображениями лиц
image_size = 64         # Размер изображений (64x64)
latent_dim = 128        # Размерность скрытого пространства
batch_size = 256        # Размер пакета (batch)
num_workers = 18        # Количество потоков для загрузки данных
epochs = 501            # Количество эпох обучения
lr_g = 0.0002          # Learning rate для генератора
lr_d = 0.00005         # Learning rate для дискриминатора
beta1 = 0.5             # Beta1 для оптимизатора Adam
beta2 = 0.999           # Beta2 для оптимизатора Adam
save_interval = 10      # Интервал сохранения моделей (каждые N эпох)

# --- Определение устройства ---
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
#print(f"Using device: {device}")

# --- Трансформации ---
transform = transforms.Compose([
    transforms.Resize(image_size),
    transforms.CenterCrop(image_size),
    transforms.ToTensor(),
    transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))
])

# --- Dataset и DataLoader ---
class FacesDataset(Dataset):
    def __init__(self, data_dir, transform=None):
        self.data_dir = data_dir
        self.image_paths = []

        # Проходим по всем поддиректориям в data_dir
        for subdir in os.listdir(data_dir):
            subdir_path = os.path.join(data_dir, subdir)
            # Проверяем, что это директория
            if os.path.isdir(subdir_path):
                print(f"Searching for images in subdirectory: {subdir}")
                # Проходим по всем файлам в поддиректории
                for filename in os.listdir(subdir_path):
                    if filename.endswith(('.jpg', '.jpeg', '.png')):
                        image_path = os.path.join(subdir_path, filename)
                        self.image_paths.append(image_path)
                        #print(f"Found image: {image_path}") # Optional: Print each image found

        print(f"Found {len(self.image_paths)} images in total.")
        self.transform = transform

    def __len__(self):
        return len(self.image_paths)

    def __getitem__(self, idx):
        image_path = self.image_paths[idx]
        image = Image.open(image_path).convert('RGB')
        if self.transform:
            image = self.transform(image)
        return image, 0
    

# --- Генератор ---
class Generator(nn.Module):
    def __init__(self, latent_dim):
        super(Generator, self).__init__()
        self.model = nn.Sequential(
            nn.ConvTranspose2d(latent_dim, 512, kernel_size=4, stride=1, padding=0, bias=False),
            nn.BatchNorm2d(512),
            nn.ReLU(True),
            nn.ConvTranspose2d(512, 256, kernel_size=4, stride=2, padding=1, bias=False),
            nn.BatchNorm2d(256),
            nn.ReLU(True),
            nn.ConvTranspose2d(256, 128, kernel_size=4, stride=2, padding=1, bias=False),
            nn.BatchNorm2d(128),
            nn.ReLU(True),
            nn.ConvTranspose2d(128, 64, kernel_size=4, stride=2, padding=1, bias=False),
            nn.BatchNorm2d(64),
            nn.ReLU(True),
            nn.ConvTranspose2d(64, 3, kernel_size=4, stride=2, padding=1, bias=False),
            nn.Tanh()
        )

    def forward(self, z):
        return self.model(z)

# --- Дискриминатор ---
class Discriminator(nn.Module):
    def __init__(self):
        super(Discriminator, self).__init__()
        self.model = nn.Sequential(
            nn.Conv2d(3, 64, kernel_size=4, stride=2, padding=1, bias=False),
            nn.LeakyReLU(0.2, inplace=True),
            nn.Conv2d(64, 128, kernel_size=4, stride=2, padding=1, bias=False),
            nn.BatchNorm2d(128),
            nn.LeakyReLU(0.2, inplace=True),
            nn.Conv2d(128, 256, kernel_size=4, stride=2, padding=1, bias=False),
            nn.BatchNorm2d(256),
            nn.LeakyReLU(0.2, inplace=True),
            nn.Conv2d(256, 512, kernel_size=4, stride=2, padding=1, bias=False),
            nn.BatchNorm2d(512),
            nn.LeakyReLU(0.2, inplace=True),
            nn.Conv2d(512, 1, kernel_size=4, stride=1, padding=0, bias=False),
            nn.Flatten()
        ) # Удаляем сигмоиду

    def forward(self, img):
        return self.model(img)
   

# --- Функция обучения ---
def fit(model, criterion, epochs, lr_g, lr_d, save_interval, start_idx=1):
    model["discriminator"].train()
    model["generator"].train()
    torch.cuda.empty_cache()

    # Оптимизаторы с разными learning rates
    optimizer_G = optim.Adam(model["generator"].parameters(), lr=lr_g, betas=(beta1, beta2))
    optimizer_D = optim.Adam(model["discriminator"].parameters(), lr=lr_d, betas=(beta1, beta2))

    # Mixed precision training
    scaler = torch.cuda.amp.GradScaler()

    # Main Training Loop
    for epoch in range(epochs):
        loss_d_per_epoch = []
        loss_g_per_epoch = []
        real_score_per_epoch = []
        fake_score_per_epoch = []

        train_dl = DataLoader(train_ds, batch_size, shuffle=True, num_workers=num_workers, pin_memory=True)

        for real_images, _ in tqdm(train_dl, desc=f"Epoch [{epoch+1}/{epochs}]"):
            real_images = real_images.to(device)

            # Train Discriminator
            optimizer_D.zero_grad()

            with torch.cuda.amp.autocast():
                real_preds = model["discriminator"](real_images)
                real_targets = torch.ones(real_images.size(0), 1, device=device)
                real_loss = criterion["discriminator"](real_preds, real_targets)
                cur_real_score = torch.mean(real_preds).item()

                # Generate fake images
                latent = torch.randn(batch_size, latent_dim, 1, 1, device=device)
                fake_images = model["generator"](latent)

                # Pass fake images through discriminator
                fake_preds = model["discriminator"](fake_images.detach())
                fake_targets = torch.zeros(fake_images.size(0), 1, device=device)
                fake_loss = criterion["discriminator"](fake_preds, fake_targets)
                cur_fake_score = torch.mean(fake_preds).item()

                loss_d = real_loss + fake_loss

            scaler.scale(loss_d).backward()
            scaler.step(optimizer_D)
            scaler.update()

            loss_d_per_epoch.append(loss_d.item())
            real_score_per_epoch.append(cur_real_score)
            fake_score_per_epoch.append(cur_fake_score)

            # Train Generator
            optimizer_G.zero_grad()

            with torch.cuda.amp.autocast():
                latent = torch.randn(batch_size, latent_dim, 1, 1, device=device)
                fake_images = model["generator"](latent)

                preds = model["discriminator"](fake_images)
                targets = torch.ones(batch_size, 1, device=device)
                loss_g = criterion["generator"](preds, targets)

            scaler.scale(loss_g).backward()
            scaler.step(optimizer_G)
            scaler.update()

            loss_g_per_epoch.append(loss_g.item())

        # Metrics
        losses_g_mean = sum(loss_g_per_epoch) / len(loss_g_per_epoch)
        losses_d_mean = sum(loss_d_per_epoch) / len(loss_d_per_epoch)
        real_score_mean = sum(real_score_per_epoch) / len(real_score_per_epoch)
        fake_score_mean = sum(fake_score_per_epoch) / len(fake_score_per_epoch)

        # Log the metrics
        print(f"Epoch [{epoch+1}/{epochs}], loss_g: {losses_g_mean:.4f}, loss_d: {losses_d_mean:.4f}, real_score: {real_score_mean:.4f}, fake_score: {fake_score_mean:.4f}")

        # Save the model
        if epoch % save_interval == 0:
            torch.save(model["generator"].state_dict(), f'generator_epoch_{epoch+1}.pth')
            torch.save(model["discriminator"].state_dict(), f'discriminator_epoch_{epoch+1}.pth')
            print(f"Saved models at epoch {epoch+1}")

    print('Finished Training')

# --- Функция для генерации и отображения изображений ---
def generate_and_show(generator_path, latent_dim, device, num_images=64):
    # 1. Создать экземпляр генератора (с той же архитектурой, что и при обучении)
    generator = Generator(latent_dim).to(device)
    # 2. Загрузить веса
    generator.load_state_dict(torch.load(generator_path, map_location=device))
    generator.eval()  # Важно: перевести генератор в режим оценки

    # 3. Сгенерировать случайные векторы шума
    with torch.no_grad():
        noise = torch.randn(num_images, latent_dim, 1, 1, device=device)
        # 4. Сгенерировать изображения
        generated_images = generator(noise)

    # 5. Отобразить сгенерированные изображения (или сохранить их)
    generated_images = (generated_images + 1) / 2.0  # Денормализация

    # Отображение с помощью matplotlib
    plt.figure(figsize=(8, 8))
    plt.axis("off")
    plt.title("Generated Images")
    plt.imshow(vutils.make_grid(generated_images.cpu(), padding=2, normalize=True).permute((1, 2, 0)))
    plt.show()

# --- Основной код ---
if __name__ == '__main__':
    # 1. Создание dataset и dataloader
    train_ds = FacesDataset(DATA_DIR, transform=transform)

    # 2. Создание моделей
    generator = Generator(latent_dim).to(device)
    discriminator = Discriminator().to(device)

    # 3. Определение функций потерь
    criterion = {
       "discriminator": nn.BCEWithLogitsLoss(),
       "generator": nn.BCEWithLogitsLoss()
    }

    # 4. Объединение моделей в словарь
    model = {
        "generator": generator,
        "discriminator": discriminator
    }

    # 5. Обучение
    fit(model, criterion, epochs, lr_g, lr_d, save_interval)

    # 6. Генерация и отображение изображений (после обучения)
    generator_path = 'generator_epoch_501.pth'  # Или выбери другую эпоху
    generate_and_show(generator_path, latent_dim, device)
