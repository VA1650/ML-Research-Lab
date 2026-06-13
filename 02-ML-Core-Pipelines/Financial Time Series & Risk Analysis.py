import pandas as pd
import numpy as np
import logging
import sys

# Настройка логирования
logging.basicConfig(level=logging.INFO, format='%(levelname)s: %(message)s')

def calculate_var_threshold(train_series: pd.Series, confidence: float = 0.95) -> float:
    """
    Вычисляет порог Value at Risk (VaR) методом исторического моделирования.
    """
    if train_series.empty:
        raise ValueError("Входные данные пусты.")
    return float(np.percentile(train_series, confidence * 100))

def run_risk_analysis(file_path: str):
    try:
        # Загрузка данных
        data = pd.read_excel(file_path)
        if data.shape[1] < 2:
            raise ValueError("Файл должен содержать минимум 2 колонки.")
            
        # Разделение по временной шкале (60/40)
        split_idx = int(len(data) * 0.6)
        train = data.iloc[:split_idx]
        test = data.iloc[split_idx:]
        
        # Анализируем вторую колонку (BTC/RUB или аналог)
        target_col = data.columns[1]
        
        # 1. Расчет исторического порога на train
        threshold_val = calculate_var_threshold(train[target_col])
        
        # 2. Проверка на тесте
        exceeded = test[test[target_col] > threshold_val]
        
        logging.info(f"Анализ завершен успешно.")
        print(f"--- Результаты анализа риска ---")
        print(f"Порог VaR (95%): {threshold_val:.2f}")
        print(f"Тестовая выборка: {len(test)} записей")
        print(f"Количество пробитий порога: {len(exceeded)}")
        print(f"Процент пробитий: {(len(exceeded)/len(test))*100:.2f}%")
        
    except FileNotFoundError:
        logging.error(f"Файл {file_path} не найден.")
    except Exception as e:
        logging.error(f"Критическая ошибка: {e}")

if __name__ == "__main__":
    # Имя файла можно вынести в конфиг
    run_risk_analysis('1.xlsx')