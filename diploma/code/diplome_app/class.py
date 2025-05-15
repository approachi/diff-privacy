import pandas as pd
import numpy as np
import math

import seaborn as sns
import matplotlib.pyplot as plt

from scipy import stats
import numpy as np
import pandas as pd
from sklearn.preprocessing import LabelEncoder, KBinsDiscretizer
from pgmpy.models import BayesianNetwork
from pgmpy.estimators import MaximumLikelihoodEstimator, HillClimbSearch, BDsScore
from pgmpy.factors.discrete import TabularCPD
from typing import Dict, List, Tuple
import math

import lightgbm as lgb
import networkx as nx
from feature_engine import encoding as ce



from sklearn.metrics import f1_score, fbeta_score, roc_auc_score, roc_curve, auc, confusion_matrix, precision_score, accuracy_score, recall_score, mean_squared_error, mean_absolute_percentage_error, mean_absolute_error
from sklearn.model_selection import train_test_split, GridSearchCV, RepeatedStratifiedKFold, cross_validate
from sklearn.ensemble import RandomForestClassifier, RandomForestRegressor
from sklearn.linear_model import LogisticRegression, LinearRegression
from sklearn.preprocessing import StandardScaler, KBinsDiscretizer, LabelEncoder
from sklearn.pipeline import Pipeline
from sklearn.datasets import load_breast_cancer
from sklearn.impute import SimpleImputer
from sklearn.neighbors import KNeighborsClassifier
from sklearn.tree import DecisionTreeClassifier

import xgboost as xgb

import torch

from sdv.single_table import CTGANSynthesizer
from sdv.datasets.demo import get_available_demos, download_demo
from sdv.evaluation.single_table import run_diagnostic
from sdv.metadata import SingleTableMetadata

# import sys
# sys.path.append('./BAMT')

# import bamt.preprocessors as pp
# import bamt.networks as nets

from pgmpy.models import BayesianNetwork
from pgmpy.estimators import MaximumLikelihoodEstimator, HillClimbSearch, BicScore
from pgmpy.estimators import K2Score, BDsScore
from pgmpy.sampling import BayesianModelSampling


import warnings
warnings.simplefilter('ignore')

CTGAN_COLOR = '#FF00FF'
BAMT_COLOR = '#1E90FF'
REAL_COLOR = '#006400'

def lgbm_predict(xx_train, yy_train, xx_test, yy_test, categorical_features, reg=False):
    for col in categorical_features:
        xx_train[col] = xx_train[col].astype('category')
        xx_test[col] = xx_test[col].astype('category')
    if not reg:
        model = lgb.LGBMClassifier(random_state=42, categorical_feature=categorical_features)
        model.fit(xx_train, yy_train)
        y_pred_test = model.predict_proba(xx_test)[:, 1]
    else:
        model = lgb.LGBMRegressor(random_state=42, categorical_feature=categorical_features)
        model.fit(xx_train, yy_train)
        y_pred_test = model.predict(xx_test)
    return y_pred_test

def xgb_predict(xx_train, yy_train, xx_test, yy_test, categorical_features):
    for col in categorical_features:
        xx_train[col] = xx_train[col].astype('category')
        xx_test[col] = xx_test[col].astype('category')

    dtrain = xgb.DMatrix(xx_train, label=yy_train, enable_categorical=True)
    dtest = xgb.DMatrix(xx_test, enable_categorical=True)

    params = {
        'objective': 'reg:squarederror',
        'random_state': 42,
    }

    model = xgb.train(params, dtrain)
    y_pred_test = model.predict(dtest)

    return y_pred_test

def fit_predict(clf, xx_train, yy_train, xx_test, yy_test, reg=False):
    clf.fit(xx_train,yy_train)
    if not reg:
        y_pred_test = clf.predict_proba(xx_test)[:, 1]
    else:
        y_pred_test = clf.predict(xx_test)
    return y_pred_test

def remove_outliers(df, exclude_l=[]):
    # Только для числовых столбцов
    numeric_columns = df.select_dtypes(include=['float64', 'int64']).columns
    if len(exclude_l):
        numeric_columns = numeric_columns.drop(exclude_l)
    print(numeric_columns)
    
    # Для каждого столбца вычисляем Q1 (25-й процентиль) и Q3 (75-й процентиль)
    Q1 = df[numeric_columns].quantile(0.25)
    Q3 = df[numeric_columns].quantile(0.75)
    
    # Вычисляем межквартильный размах (IQR)
    IQR = Q3 - Q1
    
    # Определяем границы для удаления выбросов
    lower_bound = Q1 - 1.5 * IQR
    upper_bound = Q3 + 1.5 * IQR
    
    # Фильтруем выбросы
    df_cleaned = df[~((df[numeric_columns] < lower_bound) | (df[numeric_columns] > upper_bound)).any(axis=1)]
    
    return df_cleaned

color_list = [REAL_COLOR, CTGAN_COLOR, BAMT_COLOR, "red", "black", "purple"]
def plot_roc_curve(y_all):
    fig = plt.figure(figsize=(12, 8))
    lw = 2

    for i, y_label in enumerate(y_all):
        y_true, y_pred = np.array(y_all[y_label][0]), np.array(y_all[y_label][1])

        [fpr, tpr, tresholds] = roc_curve(y_true, y_pred)
        metric_value = auc(fpr, tpr)
        plt.plot(
            fpr,
            tpr,
            color=color_list[i],
            lw=lw,
            label=f"AUC {y_label} {round(metric_value, 3)}",
        )

    plt.plot([0, 1], [0, 1], color="navy", lw=lw, linestyle="--")
    plt.title("Roc curve", fontsize=18)
    plt.xlabel("False Positive Rate", fontsize=18)
    plt.ylabel("True Positive Rate", fontsize=18)
    plt.grid(color="gray", linestyle="--", linewidth=0.5)
    plt.legend(loc="lower right", fontsize=14)
    plt.show()


data = pd.read_csv('./data/BankChurners.csv')

churn_map = {'Existing Customer': 0,
            'Attrited Customer': 1}

data['target'] = data['Attrition_Flag'].map(churn_map)

data.drop('Attrition_Flag', axis=1, inplace=True)

data.drop(['CLIENTNUM',
        'Naive_Bayes_Classifier_Attrition_Flag_Card_Category_Contacts_Count_12_mon_Dependent_count_Education_Level_Months_Inactive_12_mon_1',
     'Naive_Bayes_Classifier_Attrition_Flag_Card_Category_Contacts_Count_12_mon_Dependent_count_Education_Level_Months_Inactive_12_mon_2'],
        axis=1, inplace=True)

data[data.columns[15:16]] = data[data.columns[15:16]].astype(float)

# data.head()

data = data[['Gender', 'Dependent_count', 'Marital_Status', 'Income_Category', 'Card_Category', 
             'Total_Relationship_Count', 'Credit_Limit', 'Avg_Open_To_Buy', 'Total_Trans_Amt', 
             'Total_Trans_Ct', 'Customer_Age', 'Months_on_book', 
             'Total_Amt_Chng_Q4_Q1', 'target']]

X_train, X_test, y_train, y_test = train_test_split(data.drop(['target'], axis=1),
                                                    data['target'], test_size=0.1,
                                                    shuffle=True, stratify=data['target'],
                                                    random_state=42)

print(X_train.shape, X_test.shape)

real_data = X_train.copy()
real_data['target'] = y_train
categorical_features = list(real_data.loc[:, real_data.dtypes == object].columns)

print(categorical_features)

for col in categorical_features:
    print(col, '-', real_data[col].nunique())

real_data.dtypes.value_counts()

from sklearn.preprocessing import LabelEncoder
from diffprivlib.models import GaussianNB

# Копируем данные, чтобы не изменять оригинальные
X_train_encoded = X_train.copy()
X_test_encoded = X_test.copy()

# Применяем LabelEncoder к каждому категориальному столбцу
label_encoders = {}
for column in X_train_encoded.select_dtypes(include=['object', 'category']).columns:
    le = LabelEncoder()
    X_train_encoded[column] = le.fit_transform(X_train_encoded[column])
    X_test_encoded[column] = le.transform(X_test_encoded[column])
    label_encoders[column] = le  # Сохраняем encoder для обратного преобразования

# # Теперь обучаем модель с дифференциальной приватностью
# clf = GaussianNB(epsilon=0.1)
# print(clf.fit(X_train_encoded, y_train))

# def SyntheticBN(real_data):
#     label_encoders = {}
#     real_data_enc = real_data.copy()

#     for column in real_data_enc.select_dtypes(include=['object', 'category']).columns:
#         le = LabelEncoder()
#         real_data_enc[column] = le.fit_transform(real_data_enc[column])
#         label_encoders[column] = le

#     real_data_enc = real_data_enc.dropna()
#     print(real_data_enc)

#     cleaned_real_data = remove_outliers(real_data_enc, exclude_l=['target', 'Card_Category'])

#     print(f"Размер данных до очистки: {real_data_enc.shape}")
#     print(f"Размер данных после очистки: {cleaned_real_data.shape}")

#     # Выбираем непрерывные переменные
#     continuous_columns = cleaned_real_data.select_dtypes(include=['float64', 'int64']).columns
#     # Дискретизация с использованием равных интервалов (по умолчанию — 5 интервалов)
#     discretizer = KBinsDiscretizer(n_bins=5, encode='ordinal', strategy='uniform')
#     # Применяем дискретизацию к непрерывным переменным
#     cleaned_real_data[continuous_columns] = discretizer.fit_transform(cleaned_real_data[continuous_columns])

#     hc = HillClimbSearch(cleaned_real_data)
#     # best_model = hc.estimate(scoring_method=BicScore(real_data))
#     best_model = hc.estimate(scoring_method=BDsScore(cleaned_real_data)) #  K2Score

#     model = BayesianNetwork(best_model.edges())
#     model.fit(real_data_enc, estimator=MaximumLikelihoodEstimator)

#     #Генерация новых синтетических данных
#     sampler = BayesianModelSampling(model)

#     synthetic_samples = sampler.forward_sample(size=real_data_enc.shape[0])

#     synthetic_samples_df = pd.DataFrame(synthetic_samples)

#     synthetic_samples_df

#     cleared_synthetic_samples_df = remove_outliers(synthetic_samples_df)
#     cleared_synthetic_samples_df.shape

#     # Декодируем обратно категориальные переменные
#     for column, le in label_encoders.items():
#         synthetic_samples_df[column] = le.inverse_transform(synthetic_samples_df[column])

#     synthetic_samples_df

#     synthetic_samples_df.to_csv('./synth_data/synthetic_data_BN_cls.csv', index=False)

# SyntheticBN(real_data=real_data)

# class PrivateBayesianSynthesizer:
#     """
#     Класс для генерации синтетических данных с помощью Байесовской сети
#     с применением дифференциальной приватности.
#     """
    
#     def __init__(
#         self,
#         epsilon: float = 1.0,
#         n_bins: int = 5,
#         random_state: int = 42
#     ):
#         """
#         Инициализация синтезатора.
        
#         Args:
#             epsilon (float): Параметр приватности
#             n_bins (int): Количество интервалов для дискретизации
#             random_state (int): Seed для воспроизводимости
#         """
#         self.epsilon = epsilon
#         self.n_bins = n_bins
#         self.random_state = random_state
#         self.label_encoders = {}
#         self.discretizer = None
#         self.model = None
#         np.random.seed(random_state)
        
#     def _add_laplace_noise(self, counts: np.ndarray, sensitivity: float = 1.0) -> np.ndarray:
#         """
#         Добавление шума Лапласа для обеспечения дифференциальной приватности.
        
#         Args:
#             counts (np.ndarray): Массив значений
#             sensitivity (float): Чувствительность запроса
            
#         Returns:
#             np.ndarray: Массив с добавленным шумом
#         """
#         scale = sensitivity / self.epsilon
#         noise = np.random.laplace(0, scale, counts.shape)
#         noisy_counts = counts + noise
#         return np.maximum(noisy_counts, 0)  # Обеспечиваем неотрицательность

#     def _preprocess_data(
#         self,
#         data: pd.DataFrame,
#         continuous_columns: List[str]
#     ) -> pd.DataFrame:
#         """
#         Предобработка данных: кодирование категориальных и дискретизация непрерывных признаков.
        
#         Args:
#             data (pd.DataFrame): Исходные данные
#             continuous_columns (List[str]): Список непрерывных признаков
            
#         Returns:
#             pd.DataFrame: Предобработанные данные
#         """
#         processed_data = data.copy()
        
#         # Кодирование категориальных признаков
#         categorical_columns = processed_data.select_dtypes(
#             include=['object', 'category']
#         ).columns
        
#         for column in categorical_columns:
#             if column not in self.label_encoders:
#                 self.label_encoders[column] = LabelEncoder()
#                 processed_data[column] = self.label_encoders[column].fit_transform(
#                     processed_data[column].astype(str)
#                 )
#             else:
#                 processed_data[column] = self.label_encoders[column].transform(
#                     processed_data[column].astype(str)
#                 )
        
#         # Дискретизация непрерывных признаков
#         if continuous_columns:
#             if self.discretizer is None:
#                 self.discretizer = KBinsDiscretizer(
#                     n_bins=self.n_bins,
#                     encode='ordinal',
#                     strategy='uniform'
#                 )
#                 processed_data[continuous_columns] = self.discretizer.fit_transform(
#                     processed_data[continuous_columns]
#                 )
#             else:
#                 processed_data[continuous_columns] = self.discretizer.transform(
#                     processed_data[continuous_columns]
#                 )
        
#         # Убеждаемся, что все значения целочисленные
#         for column in processed_data.columns:
#             processed_data[column] = processed_data[column].astype(int)
        
#         return processed_data

#     def _private_contingency_table(
#         self,
#         data: pd.DataFrame,
#         variables: List[str]
#     ) -> np.ndarray:
#         """
#         Создание приватной таблицы сопряженности.
        
#         Args:
#             data (pd.DataFrame): Данные
#             variables (List[str]): Список переменных
            
#         Returns:
#             np.ndarray: Приватная таблица сопряженности
#         """
#         # Создаем таблицу сопряженности
#         contingency = pd.crosstab(
#             [data[v] for v in variables[:-1]],
#             data[variables[-1]]
#         ).values
        
#         # Добавляем шум Лапласа
#         private_contingency = self._add_laplace_noise(contingency)
        
#         return private_contingency

#     def _create_private_cpd(
#         self,
#         data: pd.DataFrame,
#         node: str,
#         parents: List[str]
#     ) -> TabularCPD:
#         """
#         Создание приватной таблицы условных вероятностей.
        
#         Args:
#             data (pd.DataFrame): Данные
#             node (str): Целевая переменная
#             parents (List[str]): Список родительских переменных
            
#         Returns:
#             TabularCPD: Таблица условных вероятностей
#         """
#         # Получаем количество уникальных значений для целевой переменной
#         variable_values = sorted(data[node].unique())
#         variable_card = len(variable_values)
        
#         if not parents:
#             # Если нет родительских переменных, создаем простое распределение
#             counts = np.zeros(variable_card)
#             for idx, val in enumerate(variable_values):
#                 counts[idx] = np.sum(data[node] == val)
            
#             # Добавляем шум и нормализуем
#             counts = self._add_laplace_noise(counts)
#             probabilities = (counts / (counts.sum() + 1e-10)).reshape(-1, 1)
            
#             return TabularCPD(
#                 variable=node,
#                 variable_card=int(variable_card),
#                 values=probabilities,
#                 evidence=None,
#                 evidence_card=None
#             )
        
#         # Если есть родительские переменные
#         evidence_card = []
#         for parent in parents:
#             parent_values = sorted(data[parent].unique())
#             evidence_card.append(len(parent_values))
        
#         # Создаем все возможные комбинации значений родительских переменных
#         parent_values = [range(card) for card in evidence_card]
#         combinations = np.array(np.meshgrid(*parent_values)).T.reshape(-1, len(parents))
        
#         # Инициализируем массив для вероятностей
#         total_combinations = np.prod(evidence_card)
#         probabilities = np.zeros((variable_card, int(total_combinations)))
        
#         # Заполняем вероятности для существующих комбинаций
#         for i, comb in enumerate(combinations):
#             mask = np.ones(len(data), dtype=bool)
#             for parent, value in zip(parents, comb):
#                 mask &= (data[parent] == value)
            
#             if mask.any():
#                 # Получаем значения для текущей комбинации
#                 current_values = data.loc[mask, node].values
#                 counts = np.zeros(variable_card)
                
#                 # Подсчитываем частоты для каждого значения
#                 for idx, val in enumerate(variable_values):
#                     counts[idx] = np.sum(current_values == val)
                
#                 # Добавляем шум и нормализуем
#                 counts = self._add_laplace_noise(counts)
#                 probabilities[:, i] = counts / (counts.sum() + 1e-10)
#             else:
#                 # Если комбинация не встречается, используем равномерное распределение
#                 probabilities[:, i] = 1.0 / variable_card
        
#         # Проверяем, что размерности корректны
#         expected_shape = (variable_card, int(np.prod(evidence_card)))
#         assert probabilities.shape == expected_shape, \
#             f"Expected shape {expected_shape}, got {probabilities.shape}"
        
#         return TabularCPD(
#             variable=node,
#             variable_card=int(variable_card),
#             values=probabilities,
#             evidence=parents,
#             evidence_card=evidence_card
#         )
#     def fit(
#         self,
#         data: pd.DataFrame,
#         continuous_columns: List[str] = None
#     ) -> None:
#         """
#         Обучение модели на данных.
        
#         Args:
#             data (pd.DataFrame): Исходные данные
#             continuous_columns (List[str]): Список непрерывных признаков
#         """
#         # Предобработка данных
#         processed_data = self._preprocess_data(data, continuous_columns)
        
#         # Поиск структуры Байесовской сети
#         hc = HillClimbSearch(processed_data)
#         best_model = hc.estimate(
#             scoring_method=BDsScore(processed_data)
#         )
        
#         # Создание и обучение модели с приватными CPD
#         self.model = BayesianNetwork(best_model.edges())
        
#         # Для каждой вершины создаем приватную CPD
#         for node in self.model.nodes():
#             parents = list(self.model.get_parents(node))
#             cpd = self._create_private_cpd(processed_data, node, parents)
#             self.model.add_cpds(cpd)

#     def generate(self, n_samples: int) -> pd.DataFrame:
#         """
#         Генерация синтетических данных.
        
#         Args:
#             n_samples (int): Количество генерируемых образцов
            
#         Returns:
#             pd.DataFrame: Сгенерированные данные
#         """
#         if not self.model:
#             raise ValueError("Модель не обучена. Сначала выполните метод fit().")
        
#         # Генерация синтетических данных
#         synthetic_data = self.model.simulate(n_samples=n_samples)
        
#         # Обратное преобразование категориальных признаков
#         for column, le in self.label_encoders.items():
#             synthetic_data[column] = le.inverse_transform(synthetic_data[column])
        
#         return synthetic_data
    
# # Инициализация синтезатора с дифференциальной приватностью
# private_synthesizer = PrivateBayesianSynthesizer(
#     epsilon=1.0,  # параметр приватности
#     n_bins=5,     # количество интервалов для дискретизации
#     random_state=42
# )

# # Определение непрерывных признаков
# continuous_columns = real_data.select_dtypes(include=['float64', 'int64']).columns.tolist()

# # Обучение модели
# private_synthesizer.fit(real_data, continuous_columns=continuous_columns)

# # Генерация синтетических данных
# synthetic_samples_df = private_synthesizer.generate(n_samples=real_data.shape[0])

# # Сохранение синтетических данных
# synthetic_samples_df.to_csv('./synth_data/private_synthetic_data_BN.csv', index=False)

class DataVisualizer:
    """
    Класс для визуализации и сравнения реальных и синтетических данных
    """
    def __init__(self, real_data, synthetic_data, figsize=(15, 10)):
        self.real_data = real_data
        self.synthetic_data = synthetic_data
        self.figsize = figsize
        
    def plot_numerical_distributions(self, columns=None):
        """
        Построение распределений числовых признаков
        """
        if columns is None:
            columns = self.real_data.select_dtypes(include=['float64', 'int64']).columns
        
        n_cols = 2
        n_rows = (len(columns) + 1) // 2
        
        fig, axes = plt.subplots(n_rows, n_cols, figsize=(15, 5*n_rows))
        axes = axes.flatten()
        
        for idx, col in enumerate(columns):
            # Построение KDE для реальных данных
            sns.kdeplot(data=self.real_data, x=col, ax=axes[idx], 
                       label='Real', color=REAL_COLOR)
            
            # Построение KDE для синтетических данных
            sns.kdeplot(data=self.synthetic_data, x=col, ax=axes[idx], 
                       label='Synthetic', color=CTGAN_COLOR)
            
            axes[idx].set_title(f'Distribution of {col}')
            axes[idx].legend()
            
            # Добавление статистик
            ks_stat, p_value = stats.ks_2samp(
                self.real_data[col].dropna(),
                self.synthetic_data[col].dropna()
            )
            axes[idx].text(0.05, 0.95, 
                         f'KS test p-value: {p_value:.4f}\nKS statistic: {ks_stat:.4f}',
                         transform=axes[idx].transAxes,
                         verticalalignment='top',
                         bbox=dict(boxstyle='round', facecolor='white', alpha=0.8))
        
        # Удаление пустых подграфиков
        for idx in range(len(columns), len(axes)):
            fig.delaxes(axes[idx])
            
        plt.tight_layout()
        plt.show()
        
    def plot_categorical_distributions(self, columns=None):
        """
        Построение распределений категориальных признаков
        """
        if columns is None:
            columns = self.real_data.select_dtypes(include=['object', 'category']).columns
        
        n_cols = 2
        n_rows = (len(columns) + 1) // 2
        
        fig, axes = plt.subplots(n_rows, n_cols, figsize=(15, 5*n_rows))
        axes = axes.flatten()
        
        for idx, col in enumerate(columns):
            # Подсчет процентных долей для реальных данных
            real_props = self.real_data[col].value_counts(normalize=True)
            synth_props = self.synthetic_data[col].value_counts(normalize=True)
            
            # Создание DataFrame для построения
            plot_data = pd.DataFrame({
                'Real': real_props,
                'Synthetic': synth_props
            }).fillna(0)
            
            # Построение сгруппированных столбчатых диаграмм
            plot_data.plot(kind='bar', ax=axes[idx], width=0.8)
            
            axes[idx].set_title(f'Distribution of {col}')
            axes[idx].set_ylabel('Proportion')
            axes[idx].tick_params(axis='x', rotation=45)
            
            # Добавление хи-квадрат теста
            chi2, p_value = stats.chi2_contingency(
                pd.crosstab(self.real_data[col], 'Real').join(
                    pd.crosstab(self.synthetic_data[col], 'Synthetic')
                ).fillna(0)
            )[:2]
            
            axes[idx].text(0.05, 0.95,
                         f'Chi2 test p-value: {p_value:.4f}\nChi2 statistic: {chi2:.4f}',
                         transform=axes[idx].transAxes,
                         verticalalignment='top',
                         bbox=dict(boxstyle='round', facecolor='white', alpha=0.8))
        
        # Удаление пустых подграфиков
        for idx in range(len(columns), len(axes)):
            fig.delaxes(axes[idx])
            
        plt.tight_layout()
        plt.show()
        
    def plot_correlation_comparison(self):
        """
        Построение сравнения корреляционных матриц
        """
        # Получение числовых столбцов
        numeric_cols = self.real_data.select_dtypes(include=['float64', 'int64']).columns
        
        # Вычисление корреляционных матриц
        real_corr = self.real_data[numeric_cols].corr()
        synth_corr = self.synthetic_data[numeric_cols].corr()
        
        # Построение графиков
        fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(20, 8))
        
        # Тепловая карта для реальных данных
        sns.heatmap(real_corr, annot=True, cmap='coolwarm', center=0, ax=ax1)
        ax1.set_title('Real Data Correlations')
        
        # Тепловая карта для синтетических данных
        sns.heatmap(synth_corr, annot=True, cmap='coolwarm', center=0, ax=ax2)
        ax2.set_title('Synthetic Data Correlations')
        
        plt.tight_layout()
        plt.show()
        
        # Построение разницы корреляций
        plt.figure(figsize=(10, 8))
        sns.heatmap(real_corr - synth_corr, annot=True, cmap='coolwarm', center=0)
        plt.title('Correlation Difference (Real - Synthetic)')
        plt.tight_layout()
        plt.show()
        
    def plot_pairwise_relationships(self, columns=None, n_samples=1000):
        """
        Построение попарных отношений между признаками
        """
        if columns is None:
            columns = self.real_data.select_dtypes(include=['float64', 'int64']).columns[:3]
        
        # Создание DataFrame с метками для источника данных
        real_sample = self.real_data[columns].sample(n=n_samples, random_state=42)
        real_sample['Data Type'] = 'Real'
        
        synth_sample = self.synthetic_data[columns].sample(n=n_samples, random_state=42)
        synth_sample['Data Type'] = 'Synthetic'
        
        combined_data = pd.concat([real_sample, synth_sample])
        
        # Построение попарных графиков
        sns.pairplot(combined_data, hue='Data Type', 
                    palette={
                        'Real': REAL_COLOR,
                        'Synthetic': CTGAN_COLOR
                    })
        plt.show()


# real_data = pd.read_csv('./synth_data/private_synthetic_data_BN.csv')
# synthetic_data = pd.read_csv('./synth_data/synthetic_data_BN_cls.csv')

# # Создание визуализатора
# visualizer = DataVisualizer(real_data, synthetic_data)

# # Построение всех графиков
# print("Распределения числовых признаков:")
# visualizer.plot_numerical_distributions()

# print("\nРаспределения категориальных признаков:")
# visualizer.plot_categorical_distributions()

# print("\nСравнение корреляций:")
# visualizer.plot_correlation_comparison()

# print("\nПопарные отношения:")
# visualizer.plot_pairwise_relationships()


def preds(synthetic_data_bamt, synthetic_data_ctgan):
    categorical_features = list(data.select_dtypes('object').columns)

    LR_pipe = Pipeline([
        ('encoding', ce.OrdinalEncoder(
        encoding_method='arbitrary', variables=categorical_features)),
        ('scaler', StandardScaler()),
        ('LR', LogisticRegression())
        ])

    RF_pipe = Pipeline([
        ('encoding', ce.OrdinalEncoder(
        encoding_method='arbitrary', variables=categorical_features)),
        ('scaler', StandardScaler()),
        ('RF', RandomForestClassifier())
        ])

    RFL_pipe = Pipeline([
        ('encoding', ce.OrdinalEncoder(
        encoding_method='ordered', variables=categorical_features)),
        ('scaler', StandardScaler()),
        ('RF', RandomForestRegressor())
        ])
    
    for col in categorical_features:
        synthetic_data_bamt[col] = synthetic_data_bamt[col].astype('category')
        synthetic_data_ctgan[col] = synthetic_data_ctgan[col].astype('category')

    right_pos_df_ctgan = pd.DataFrame()
    right_pos_df_bn = pd.DataFrame()

    for col in real_data.columns:
        right_pos_df_ctgan[col] = synthetic_data_ctgan[col]
        right_pos_df_bn[col] = synthetic_data_bamt[col]

    synthetic_data_ctgan = right_pos_df_ctgan.copy()
    synthetic_data_bamt = right_pos_df_bn.copy()

    y_pred_test_initial_lr = fit_predict(LR_pipe, X_train.copy(), y_train, X_test, y_test)
    y_pred_test_ctgan_lr = fit_predict(LR_pipe, synthetic_data_ctgan.drop(['target'], axis=1), synthetic_data_ctgan['target'], X_test, y_test)
    y_pred_test_bamt_lr = fit_predict(LR_pipe, synthetic_data_bamt.drop(['target'], axis=1), synthetic_data_bamt['target'], X_test, y_test)

    y_pred_test_initial_rf = fit_predict(RF_pipe, X_train.copy(), y_train, X_test, y_test)
    y_pred_test_ctgan_rf = fit_predict(RF_pipe, synthetic_data_ctgan.drop(['target'], axis=1), synthetic_data_ctgan['target'], X_test, y_test)
    y_pred_test_bamt_rf = fit_predict(RF_pipe, synthetic_data_bamt.drop(['target'], axis=1), synthetic_data_bamt['target'], X_test, y_test)

    y_pred_test_initial_lgbm = lgbm_predict(X_train.copy(), y_train, X_test, y_test, categorical_features)
    y_pred_test_ctgan_lgbm = lgbm_predict(synthetic_data_ctgan.drop(['target'], axis=1), synthetic_data_ctgan['target'], X_test, y_test, categorical_features)
    y_pred_test_bamt_lgbm = lgbm_predict(synthetic_data_bamt.drop(['target'], axis=1), synthetic_data_bamt['target'], X_test, y_test, categorical_features)


    df_lr = pd.DataFrame(data=[roc_auc_score(y_test,  y_pred_test_initial_lr), roc_auc_score(y_test,  y_pred_test_ctgan_lr), roc_auc_score(y_test,  y_pred_test_bamt_lr)],
                        columns = ['Test'],
                        index = ['Initial', 'CTGAN', 'BAMT'])
    df_lr['model']='LR'

    df_rf = pd.DataFrame(data=[roc_auc_score(y_test,  y_pred_test_initial_rf), roc_auc_score(y_test,  y_pred_test_ctgan_rf), roc_auc_score(y_test,  y_pred_test_bamt_rf)],
                        columns = ['Test'],
                        index = ['Initial', 'CTGAN', 'BAMT'])
    df_rf['model']='RF'

    df_lgbm = pd.DataFrame(data=[roc_auc_score(y_test,  y_pred_test_initial_lgbm), roc_auc_score(y_test,  y_pred_test_ctgan_lgbm), roc_auc_score(y_test,  y_pred_test_bamt_lgbm)],
                    columns = ['Test'],
                        index = ['Initial', 'CTGAN', 'BAMT'])
    df_lgbm['model']='Lgbm'
    df = pd.concat([df_lr, df_rf, df_lgbm])
    df.index.rename('Data', inplace=True)
    df.reset_index(inplace=True)

    # Визуализация
    fig = plt.figure(figsize=(12,8))
    sns.barplot(x='model', y='Test', hue='Data', data=df, 
                palette=['#66CDAA', '#FF99FF', '#87CEFA', '#FFA07A'])  # Добавил новый цвет для DP
    plt.plot([-0.5, 3.5], [1, 1], color="red", linestyle="--")
    plt.title('ROC AUC Score Comparison (with Differential Privacy)')
    plt.xlim([-0.5, 3.5])  # Расширил диапазон для новой модели
    plt.ylim([0,1.2])
    plt.grid()
    plt.show()

# 
# synthetic_data = p
preds(pd.read_csv('./synth_data/private_synthetic_data_BN.csv'), pd.read_csv('./synth_data/synthetic_data_BN_cls.csv'))