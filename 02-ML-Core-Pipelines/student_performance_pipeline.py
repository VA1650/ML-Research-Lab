import os
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.linear_model import Ridge # Используем Ridge (Линейная регрессия с L2-регуляризацией для стабильности)
from sklearn.multioutput import MultiOutputRegressor
from sklearn.metrics import r2_score, mean_absolute_error

class StudentPerformancePredictor:
    """
    Пайплайн множественной регрессии (Multi-output Regression) 
    для прогнозирования академических результатов студентов (Math, Reading, Writing scores).
    """
    def __init__(self):
        self.categorical_features = ['gender', 'race/ethnicity', 'parental level of education', 'lunch', 'test preparation course']
        self.target_columns = ['math score', 'reading score', 'writing score']
        self.pipeline = None

    def _build_pipeline(self) -> Pipeline:
        """Сборка сквозного конвейера с защитой от мультиколлинеарности."""
        # drop='first' удаляет избыточный столбец, ликвидируя Dummy Variable Trap
        preprocessor = ColumnTransformer(
            transformers=[
                ('cat', OneHotEncoder(drop='first', handle_unknown='ignore'), self.categorical_features)
            ],
            remainder='passthrough' # Если в датасете появятся числовые фичи, они пройдут дальше
        )

        # Оборачиваем стабильный линейный алгоритм Ridge в MultiOutputRegressor
        return Pipeline(steps=[
            ('preprocessor', preprocessor),
            ('scaler', StandardScaler(with_mean=False)), # Масштабируем разреженную матрицу после OHE
            ('regressor', MultiOutputRegressor(Ridge(alpha=1.0, random_state=42)))
        ])

    def train_and_evaluate(self, data_path: str):
        """Пайплайн обучения и многокритериальной оценки моделей."""
        if not os.path.exists(data_path):
            print(f"[ERROR] Файл {data_path} не найден.")
            return

        df = pd.read_csv(data_path)
        
        X = df.drop(columns=self.target_columns)
        y = df[self.target_columns]

        # Разделение данных
        X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=42)

        self.pipeline = self._build_pipeline()
        
        print("[INFO] Обучение Multi-output регрессионного пайплайна...")
        self.pipeline.fit(X_train, y_train)

        # Инференс вектора целей
        y_pred = self.pipeline.predict(X_test)

        print("\n=== Результаты валидации по дисциплинам ===")
        for i, target_name in enumerate(self.target_columns):
            r2 = r2_score(y_test.iloc[:, i], y_pred[:, i])
            mae = mean_absolute_error(y_test.iloc[:, i], y_pred[:, i])
            print(f"🎯 {target_name.capitalize()}:")
            print(f"   R² Score (Коэффициент детерминации): {r2:.4f}")
            print(f"   MAE (Средняя абсолютная ошибка):    {mae:.2f} баллов")

        # Продакшн-тест инференса одной строкой
        print("\n[INFO] Пример предикта для нового студента (Production Inference):")
        sample_student = X_test.iloc[[0]]
        predicted_scores = self.pipeline.predict(sample_student)[0]
        
        print(f"Входной профиль:\n{sample_student.to_string(index=False)}")
        print(f"Прогноз оценок -> Math: {predicted_scores[0]:.1f}, Reading: {predicted_scores[1]:.1f}, Writing: {predicted_scores[2]:.1f}")

if __name__ == "__main__":
    predictor = StudentPerformancePredictor()
    predictor.train_and_evaluate('exams.csv')