import os
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from scipy.stats import spearmanr

class UniversityRankingsAnalyzer:
    """
    Пайплайн эксплораторного анализа данных (EDA) мирового рейтинга вузов CWUR.
    Фокусируется на макро-анализе стран и нелинейных зависимостях метрик.
    """
    def __init__(self):
        self.selected_countries = ['Japan', 'Israel', 'Germany', 'France', 'Russia']
        # Настройка глобального стиля графиков для продакшн-отчетов
        sns.set_theme(style="whitegrid")
        plt.rcParams['figure.figsize'] = (10, 6)

    def process_and_visualize(self, data_path: str):
        if not os.path.exists(data_path):
            print(f"[ERROR] Файл {data_path} не найден.")
            return

        # 1. Загрузка и первичная фильтрация
        df = pd.read_csv(data_path)
        sub_df = df[df['country'].isin(self.selected_countries)].copy()
        
        # Дропаем дубликаты вузов, оставляя только самый свежий год исследования (например, 2015)
        latest_year = sub_df['year'].max()
        sub_df = sub_df[sub_df['year'] == latest_year]
        
        print(f"[INFO] Анализ производится по срезу за {latest_year} год. Объем выборки: {len(sub_df)} вузов.")

        # =====================================================================
        # 2. Визуализация: Распределение представительства стран в ТОП-CWUR
        # =====================================================================
        university_counts = sub_df['country'].value_counts()
        
        fig, ax = plt.subplots(1, 2, figsize=(16, 7))
        
        # Каноничная круговая диаграмма
        ax[0].pie(
            university_counts, 
            labels=university_counts.index, 
            autopct='%1.1f%%', 
            startangle=140,
            colors=sns.color_palette("Pastel1")
        )
        ax[0].set_title(f'Доля стран по числу топ-вузов в рейтинге ({latest_year} г.)', fontsize=12, fontweight='bold')

        # =====================================================================
        # 3. Визуализация: Влияние vs Качество образования (Advanced Scatter)
        # =====================================================================
        # Размер точки пропорционален итоговому скору вуза (score) — чем больше, тем детальнее
        scatter = sns.scatterplot(
            data=sub_df,
            x='influence',
            y='quality_of_education',
            hue='country',
            size='score',
            sizes=(40, 400),
            palette='Set1',
            alpha=0.8,
            ax=ax[1]
        )
        
        # Инвертируем оси, так как ранг 1 — это лучший показатель, а 1000 — худший
        ax[1].invert_xaxis()
        ax[1].invert_yaxis()
        
        ax[1].set_xlabel('Ранг влияния (Influence Rank, меньше = лучше)')
        ax[1].set_ylabel('Ранг качества образования (Quality of Education Rank, меньше = лучше)')
        ax[1].set_title('Корреляция: Влияние vs Качество Образования', fontsize=12, fontweight='bold')
        ax[1].legend(bbox_to_anchor=(1.05, 1), loc='upper left', title='Параметры')

        plt.tight_layout()
        
        # Сохраняем аналитический дашборд
        output_plot = "cwur_analytical_dashboard.png"
        plt.savefig(output_plot, dpi=300, bbox_inches='tight')
        print(f"[INFO] Дашборд успешно экспортирован в файл '{output_plot}'")
        plt.show()

        # =====================================================================
        # 4. Математический анализ зависимостей
        # =====================================================================
        corr, p_value = spearmanr(sub_df['influence'], sub_df['quality_of_education'])
        print("\n=== Статистический анализ ===")
        print(f"Ранговая корреляция Спирмена: {corr:.4f}")
        print(f"Статистическая значимость (p-value): {p_value:.4e}")
        if p_value < 0.05:
            print("[CONCLUSION] Зависимость между влиянием вуза и качеством образования статистически подтверждена.")
        else:
            print("[CONCLUSION] Корреляция не является статистически значимой.")

if __name__ == "__main__":
    # Для работы требуется файл cwurData.csv в корне
    analyzer = UniversityRankingsAnalyzer()
    analyzer.analyze_and_visualize("cwurData.csv")