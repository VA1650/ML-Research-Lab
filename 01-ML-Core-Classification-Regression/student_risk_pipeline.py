import numpy as np
import pandas as pd
from sklearn.tree import DecisionTreeClassifier, export_text
from sklearn.metrics import classification_report, confusion_matrix
from sklearn.model_selection import train_test_split

class StudentRiskSimulator:
    """
    Симулятор синтетических данных для моделирования рисков отчисления студентов.
    Используется для верификации предсказательной способности дерева решений (ID3/C4.5).
    """
    def __init__(self, n_samples: int = 1000, random_seed: int = 42):
        self.n_samples = n_samples
        np.random.seed(random_seed)
        
    def generate_dataset(self) -> pd.DataFrame:
        """Генерация непрерывных и категориальных признаков с контролируемой логикой."""
        # Симулируем реальные метрики студентов
        attendance = np.random.randint(10, 101, size=self.n_samples)      # Процент посещаемости (10% - 100%)
        past_debts = np.random.randint(0, 5, size=self.n_samples)         # Кол-во задолженностей (0 - 4)
        gpa = np.random.uniform(2.0, 5.0, size=self.n_samples)            # Средний балл (2.0 - 5.0)
        extracurricular = np.random.randint(0, 2, size=self.n_samples)    # Активность (0 или 1)
        
        df = pd.DataFrame({
            'attendance_pct': attendance,
            'failed_credits': past_debts,
            'gpa': gpa,
            'is_active_community': extracurricular
        })

        # Задаем жесткую бизнес-логику для таргета (Target Labeling)
        # Студент в зоне риска (is_at_risk = 1), если критически низкая посещаемость ИЛИ много долгов при низком GPA
        conditions = (df['attendance_pct'] < 40) | ((df['failed_credits'] >= 2) & (df['gpa'] < 3.5))
        df['is_at_risk'] = np.where(conditions, 1, 0)
        
        # Добавляем в таргет 5% случайного шума для симуляции реальных аномалий
        noise_mask = np.random.rand(self.n_samples) < 0.05
        df.loc[noise_mask, 'is_at_risk'] = 1 - df.loc[noise_mask, 'is_at_risk']
        
        return df

def run_pipeline():
    # 1. Генерация данных
    simulator = StudentRiskSimulator(n_samples=1200)
    dataset = simulator.generate_dataset()
    
    X = dataset.drop(columns=['is_at_risk'])
    y = dataset['is_at_risk']
    
    # 2. Валидация распределения классов
    print(f"[INFO] Распределение классов:\n{y.value_counts(normalize=True)}\n")

    # 3. Разделение выборки
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, stratify=y, random_state=42)

    # 4. Обучение классификатора на критерии энтропии (Информационный выигрыш / ID3)
    # Ограничиваем глубину, чтобы дерево не переобучилось под внесенный нами шум
    clf = DecisionTreeClassifier(criterion="entropy", max_depth=4, random_state=42)
    clf.fit(X_train, y_train)

    # 5. Инференс и оценка метрик
    y_pred = clf.predict(X_test)
    
    print("=== Отчет о качестве классификации ===")
    print(classification_report(y_test, y_pred, target_names=['Стабилен', 'В зоне риска']))
    
    # 6. Интерпретация модели (Feature Importance)
    print("=== Важность признаков (Feature Importance) ===")
    for name, importance in zip(X.columns, clf.feature_importances_):
        print(f"{name}: {importance:.4f}")
        
    print("\n=== Извлеченные правила принятия решений (Decision Rules) ===")
    print(export_text(clf, feature_names=list(X.columns)))

if __name__ == "__main__":
    run_pipeline()