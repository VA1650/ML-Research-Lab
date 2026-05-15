import torch
import torchvision.utils as vutils
from gan_faces import Generator  # Теперь имя файла правильное
import matplotlib.pyplot as plt

def generate_and_show(generator_path, latent_dim, device, num_images=64):
    # 1. Создать экземпляр генератора (с той же архитектурой, что и при обучении)
    generator = Generator(latent_dim).to(device)  # Не забудь указать latent_dim
    # 2. Загрузить веса
    generator.load_state_dict(torch.load(generator_path, map_location=device)) # Map location нужен, если обучал на GPU, а смотришь на CPU
    generator.eval() # Важно перевести в eval mode, чтоб BatchNorm работал правильно

    # 3. Сгенерировать случайные векторы шума (latent vectors)
    with torch.no_grad():  # Отключаем вычисление градиентов
        noise = torch.randn(num_images, latent_dim, 1, 1, device=device)
        # 4. Сгенерировать изображения
        generated_images = generator(noise)

    # 5. Отобразить сгенерированные изображения (или сохранить их)
    # Денормализация (если ты нормализовала изображения при обучении)
    generated_images = (generated_images + 1) / 2.0  # Приведение к диапазону [0, 1]

    # Отображение с помощью matplotlib
    plt.figure(figsize=(8, 8))
    plt.axis("off")
    plt.title("Generated Images")
    plt.imshow(vutils.make_grid(generated_images.cpu(), padding=2, normalize=True).permute((1, 2, 0)))
    plt.show()

# Пример использования (замени 'generator_epoch_100.pth' на свой путь)
generator_path = 'generator_epoch_201.pth'
latent_dim = 128  # Замени на размерность твоего latent space
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
generate_and_show(generator_path, latent_dim, device)
