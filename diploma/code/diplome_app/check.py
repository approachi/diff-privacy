import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from scipy import stats
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import roc_auc_score, classification_report

# Загрузка данных
real_data = pd.read_csv('./data/BankChurners.csv')
synthetic_data = pd.read_csv('./synth_data/synthetic_data_BN_cls.csv')
columns_to_ignore = ['target']  # Замените на свои колонки

# 2. Находим общие столбцы, исключая игнорируемые
common_columns = list(
    set(real_data.columns) & 
    set(synthetic_data.columns) - 
    set(columns_to_ignore)
)

print("\nАнализируемые колонки:", common_columns)

if not common_columns:
    raise ValueError("Нет общих колонок для анализа после исключения")

# 3. Основные статистики
print("\nОписательные статистики (реальные vs синтетические):")
for col in common_columns:
    if pd.api.types.is_numeric_dtype(real_data[col]):
        print(f"\nКолонка: {col}")
        print("Реальные данные:")
        print(real_data[col].describe())
        print("\nСинтетические данные:")
        print(synthetic_data[col].describe())

# 4. Визуализация распределений
for col in common_columns:
    plt.figure(figsize=(10, 5))
    
    if pd.api.types.is_numeric_dtype(real_data[col]):
        sns.kdeplot(real_data[col], label='Real', color='blue')
        sns.kdeplot(synthetic_data[col], label='Synthetic', color='orange')
        plt.title(f'Распределение для {col} (числовая)')
    else:
        real_counts = real_data[col].value_counts(normalize=True).sort_index()
        synth_counts = synthetic_data[col].value_counts(normalize=True).sort_index()
        width = 0.35
        x = np.arange(len(real_counts))
        plt.bar(x - width/2, real_counts, width, label='Real', color='blue')
        plt.bar(x + width/2, synth_counts, width, label='Synthetic', color='orange')
        plt.xticks(x, real_counts.index, rotation=45)
        plt.title(f'Частоты для {col} (категориальная)')
    
    plt.legend()
    plt.show()

# 5. Статистические тесты
print("\nСтатистические тесты:")
for col in common_columns:
    if pd.api.types.is_numeric_dtype(real_data[col]):
        ks_stat, ks_pvalue = stats.ks_2samp(real_data[col], synthetic_data[col])
        print(f"\nКолонка {col}:")
        print(f"KS-тест: статистика={ks_stat:.3f}, p-value={ks_pvalue:.3f}")
        if ks_pvalue < 0.05:
            print("❗ Значимые различия (p < 0.05)")
    else:
        contingency_table = pd.crosstab(
            index=pd.concat([real_data[col], synthetic_data[col]]),
            columns=['real']*len(real_data) + ['synthetic']*len(synthetic_data)
        )
        chi2, p, dof, expected = stats.chi2_contingency(contingency_table)
        print(f"\nКолонка {col}:")
        print(f"Chi-square: статистика={chi2:.3f}, p-value={p:.3f}")
        if p < 0.05:
            print("❗ Значимые различия (p < 0.05)")

# 6. ML-тест (только для числовых колонок)
numeric_columns = [col for col in common_columns if pd.api.types.is_numeric_dtype(real_data[col])]

if len(numeric_columns) >= 2:  # Нужно минимум 2 признака
    real_data_common = real_data[numeric_columns].copy()
    synthetic_data_common = synthetic_data[numeric_columns].copy()
    
    real_data_common['is_synthetic'] = 0
    synthetic_data_common['is_synthetic'] = 1
    
    combined_data = pd.concat([real_data_common, synthetic_data_common]).dropna()
    
    if len(combined_data) >= 20:  # Минимум 20 образцов
        X = combined_data.drop('is_synthetic', axis=1)
        y = combined_data['is_synthetic']
        
        X_train, X_test, y_train, y_test = train_test_split(
            X, y, test_size=0.3, random_state=42)
        
        model = RandomForestClassifier(random_state=42)
        model.fit(X_train, y_train)
        
        y_pred = model.predict_proba(X_test)[:, 1]
        auc_score = roc_auc_score(y_test, y_pred)
        
        print("\nРезультаты ML-теста:")
        print(f"AUC = {auc_score:.3f}")
        print("1.0 - идеальное различение, 0.5 - случайное угадывание")
        print("\nВажность признаков:")
        print(pd.DataFrame({
            'feature': X.columns,
            'importance': model.feature_importances_
        }).sort_values('importance', ascending=False))
    else:
        print("\n⚠ Слишком мало данных для ML-теста")
else:
    print("\n⚠ Недостаточно числовых колонок для ML-теста")