import os
import cv2
import numpy as np
import tensorflow as tf
from tensorflow.keras.models import Sequential, load_model
from tensorflow.keras.layers import Dense, Flatten, Rescaling
from tensorflow.keras.datasets import mnist

class DigitClassifierModel:
    """Класс для сборки, обучения и инференса нейросети."""
    def __init__(self, model_path: str = "mnist_dense_model.h5"):
        self.model_path = model_path
        self.model = None

    def build_and_train(self, epochs: int = 5):
        """Пайплайн обучения модели с нормализацией внутри графа вычислений."""
        print("[INFO] Загрузка датасета MNIST...")
        (x_train, y_train), (x_test, y_test) = mnist.load_data()
        
        # Добавляем канал для соответствия тензорному формату (28, 28, 1)
        x_train = np.expand_dims(x_train, axis=-1)
        x_test = np.expand_dims(x_test, axis=-1)

        print("[INFO] Инициализация архитектуры Keras...")
        self.model = Sequential([
            # Масштабирование признаков [0, 255] -> [0, 1] прямо внутри модели
            Rescaling(1.0 / 255, input_shape=(28, 28, 1)),
            Flatten(),
            Dense(128, activation='relu'),
            Dense(64, activation='relu'),
            Dense(10, activation='softmax')
        ])

        self.model.compile(
            optimizer='adam',
            loss='sparse_categorical_crossentropy',
            metrics=['accuracy']
        )

        print("[INFO] Старт процесса обучения...")
        self.model.fit(x_train, y_train, epochs=epochs, batch_size=64, validation_data=(x_test, y_test))
        
        self.model.save(self.model_path)
        print(f"[INFO] Модель успешно сериализована в {self.model_path}")

    def load(self):
        """Загрузка весов из файла."""
        if os.path.exists(self.model_path):
            self.model = load_model(self.model_path)
        else:
            raise FileNotFoundError(f"Файл модели {self.model_path} не найден. Сначала вызовите build_and_train().")

    def predict_digit(self, roi_gray: np.ndarray) -> int:
        """Предсказание класса для изолированной области интереса (ROI)."""
        # Векторизованный высокопроизводительный ресайз через OpenCV (без циклов)
        resized = cv2.resize(roi_gray, (28, 28), interpolation=cv2.INTER_AREA)
        tensor = np.expand_dims(resized, axis=(0, -1)) # Формируем батч (1, 28, 28, 1)
        
        preds = self.model.predict(tensor, verbose=0)
        return int(np.argmax(preds[0]))


class RealTimeVisionEngine:
    """Движок компьютерного зрения для захвата видеопотока и инференса."""
    def __init__(self, classifier: DigitClassifierModel):
        self.classifier = classifier
        self.window_name = "Real-Time Digit Classifier"

    def run(self):
        """Запуск цикла обработки видеопотока веб-камеры."""
        cap = cv2.VideoCapture(0)
        if not cap.isOpened():
            print("[ERROR] Не удалось получить доступ к видеопотоку веб-камеры.")
            return

        print("[INFO] Запуск инференса. Нарисуйте цифру в зеленом квадрате.")
        print("[INFO] Для выхода нажмите клавишу 'Q'.")

        while True:
            ret, frame = cap.read()
            if not ret:
                break

            # Инвертируем кадр для зеркального отображения
            frame = cv2.flip(frame, 1)
            h, w, _ = frame.shape

            # Задаем область интереса (ROI) в центре экрана для ввода цифры
            box_size = 200
            x1, y1 = (w - box_size) // 2, (h - box_size) // 2
            x2, y2 = x1 + box_size, y1 + box_size

            # Вырезаем область, переводим в градации серого и инвертируем цвета 
            # (MNIST обучен на белых цифрах по черному фону)
            roi = frame[y1:y2, x1:x2]
            gray = cv2.cvtColor(roi, cv2.COLOR_BGR2GRAY)
            gray = cv2.bitwise_not(gray) # Инверсия под формат MNIST

            # Получаем предсказание от нейросети
            digit = self.classifier.predict_digit(gray)

            # Отрисовка интерфейса поверх видеопотока
            cv2.rectangle(frame, (x1, y1), (x2, y2), (0, 255, 0), 2)
            cv2.putText(frame, f"Digit: {digit}", (x1, y1 - 10),
                        cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2, cv2.LINE_AA)

            cv2.imshow(self.window_name, frame)

            if cv2.waitKey(1) & 0xFF == ord('q'):
                break

        cap.release()
        cv2.destroyAllWindows()


if __name__ == "__main__":
    classifier_app = DigitClassifierModel("mnist_dense_model.h5")
    
    # Если модели еще нет — обучаем, если есть — сразу берем готовые веса
    if not os.path.exists(classifier_app.model_path):
        classifier_app.build_and_train(epochs=5)
    else:
        classifier_app.load()

    # Запуск интерактивного CV-движка
    engine = RealTimeVisionEngine(classifier_app)
    engine.run()