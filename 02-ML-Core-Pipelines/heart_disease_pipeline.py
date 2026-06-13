import os
import pandas as pd
import numpy as np
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import classification_report, roc_auc_score

class HeartDiseaseClassifier:
    """
    Пайплайн бинарной классификации рисков сердечно-сосудистых заболеваний.
    Использует сквозную фильтрацию и исключает Data Leakage.
    """
    def __init__(self):
        # Четко разделяем фичи по типам для ColumnTransformer
        self.categorical_features = ['Sex', 'ChestPainType', 'FastingBS', 'RestingECG', 'ExerciseAngina', 'ST_Slope']
        self.numerical_features = ['Age', 'RestingBP', 'Cholesterol', 'MaxHR', 'Oldpeak']
        self.pipeline = None

    def _build_pipeline(self) -> Pipeline:
        """Создание изолированного графа предобработки и модели."""
        preprocessor = ColumnTransformer(
            transformers=[
                ('num', StandardScaler(), self.numerical_features),
                ('cat', OneHotEncoder(drop='first', handle_unknown='ignore'), self.categorical_features)
            ],
            remainder='drop'
        )

        # Объединяем препроцессинг и оценщик (estimator) в один конвейер
        return Pipeline(steps=[
            ('preprocessor', preprocessor),
            ('classifier', LogisticRegression(max_iter=1000, random_state=42))
        ])

    def fit_and_evaluate(self, data_path: str):
        """Полный цикл загрузки, обучения и валидации модели."""
        if not os.path.exists(data_path):
            print(f"[ERROR] Файл {data_path} не найден.")
            return

        df = pd.read_csv(data_path)
        
        # Предполагаем, что целевая переменная — в последнем столбце (HeartDisease)
        X = df.drop(df.columns[-1], axis=1)
        y = df.iloc[:, -1]

        # Разделение данных. Стратификация обязательна для сохранения пропорций классов.
        X_train, X_test, y_train, y_test = train_test_split(
            X, y, test_size=0.2, stratify=y, random_state=42
        )

        self.pipeline = self._build_pipeline()
        
        print("[INFO] Обучение сквозного пайплайна...")
        self.pipeline.fit(X_train, y_train)

        # Инференс
        y_pred = self.pipeline.predict(X_test)
        y_proba = self.pipeline.predict_proba(X_test)[:, 1]

        # Вывод валидационных метрик
        print("\n=== Метрики качества модели (Тестовая выборка) ===")
        print(classification_report(y_test, y_pred, target_names=['Здоров', 'Группа риска']))
        print(f"ROC AUC Score: {roc_auc_score(y_test, y_proba):.4f}")

        # Демонстрация работы с сырыми данными (Production Inference Example)
        print("\n[INFO] Пример инференса на сырой строке данных:")
        sample_data = X_test.iloc[[0]]
        prediction = self.pipeline.predict(sample_data)[0]
        probability = self.pipeline.predict_proba(sample_data)[0][1]
        print(f"Входные параметры:\n{sample_data.to_string(index=False)}")
        print(f"Вердикт модели: {'Группа риска' if prediction == 1 else 'Здоров'} (Вероятность: {probability:.4f})")

if __name__ == "__main__":
    # Для запуска необходим файл heart.csv в корневом каталоге
    classifier = HeartDiseaseClassifier()
    classifier.fit_and_evaluate("heart.csv")