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
from typing import Dict, List, Tuple, Optional
import math

import lightgbm as lgb
import networkx as nx
from feature_engine import encoding as ce

import numpy as np
import pandas as pd
from sklearn.preprocessing import LabelEncoder, KBinsDiscretizer
from pgmpy.models import BayesianNetwork
from pgmpy.estimators import HillClimbSearch, BDsScore
from pgmpy.factors.discrete import TabularCPD
from typing import List, Optional
import warnings
from math import log, ceil
from scipy.optimize import fsolve

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


class PrivateBayesianSynthesizer:
    """
    Класс для генерации синтетических данных с помощью Байесовской сети
    с применением дифференциальной приватности.
    """
    
    def __init__(
        self,
        epsilon: float = 1.0,
        n_bins: int = 5,
        random_state: int = 42,
        k: Optional[int] = None,  # Если None, будет вычислен автоматически
        target_usefulness: float = 4.0
    ):
        """
        Инициализация синтезатора.
        
        Args:
            epsilon (float): Параметр приватности
            n_bins (int): Количество интервалов для дискретизации
            random_state (int): Seed для воспроизводимости
            k (Optional[int]): Макс. число родителей. Если None, вычисляется автоматически.
            target_usefulness (float): Целевая полезность для автоматического подбора k
        """
        self.epsilon = epsilon
        self.n_bins = n_bins
        self.random_state = random_state
        self.k = k
        self.target_usefulness = target_usefulness
        self.label_encoders = {}
        self.discretizer = None
        self.model = None
        np.random.seed(random_state)
    
    def _calculate_sensitivity(self, num_tuples: int, child: str, parents: List[str]) -> float:
        """
        Вычисление чувствительности для дифференциальной приватности (PrivBayes Lemma 1).
        
        Args:
            num_tuples (int): Количество записей в данных
            child (str): Дочерний атрибут
            parents (List[str]): Родительские атрибуты
            
        Returns:
            float: Значение чувствительности
        """
        attr_to_is_binary = {
            attr: len(self.label_encoders[attr].classes_) <= 2 
            for attr in [child] + parents
        }
        
        if attr_to_is_binary[child] or (len(parents) == 1 and attr_to_is_binary[parents[0]]):
            a = log(num_tuples) / num_tuples
            b = (num_tuples - 1) / num_tuples
            b_inv = num_tuples / (num_tuples - 1)
            return a + b * log(b_inv)
        else:
            a = (2 / num_tuples) * log((num_tuples + 1) / 2)
            b = (1 - 1 / num_tuples) * log(1 + 2 / (num_tuples - 1))
            return a + b
    
    def _usefulness_minus_target(
        self, 
        k: int, 
        num_attributes: int, 
        num_tuples: int
    ) -> float:
        """
        Функция полезности для автоматического подбора k (PrivBayes Lemma 3).
        """
        if k == num_attributes:
            usefulness = self.target_usefulness
        else:
            usefulness = num_tuples * self.epsilon / ((num_attributes - k) * (2 ** (k + 3)))
        return usefulness - self.target_usefulness
    
    def _calculate_k(
        self, 
        num_attributes: int, 
        num_tuples: int
    ) -> int:
        """
        Автоматический подбор k (макс. число родителей) по методу PrivBayes.
        """
        default_k = 3
        initial_usefulness = self._usefulness_minus_target(
            default_k, num_attributes, num_tuples
        )
        
        if initial_usefulness > 0:
            return default_k
        else:
            arguments = (num_attributes, num_tuples)
            warnings.filterwarnings("error")
            try:
                ans = fsolve(
                    self._usefulness_minus_target, 
                    np.array([int(num_attributes / 2)]), 
                    args=arguments
                )[0]
                ans = ceil(ans)
            except RuntimeWarning:
                print("Warning: k is not properly computed! Using default k=3")
                ans = default_k
            
            if ans < 1 or ans > num_attributes:
                ans = default_k
            return ans
        
    def _add_laplace_noise(self, counts: np.ndarray, sensitivity: float = 1.0) -> np.ndarray:
        """
        Добавление шума Лапласа для обеспечения дифференциальной приватности.
        
        Args:
            counts (np.ndarray): Массив значений
            sensitivity (float): Чувствительность запроса
            
        Returns:
            np.ndarray: Массив с добавленным шумом
        """
        scale = sensitivity / self.epsilon
        noise = np.random.laplace(0, scale, counts.shape)
        noisy_counts = counts + noise
        return np.maximum(noisy_counts, 0)  # Обеспечиваем неотрицательность

    def _preprocess_data(
        self,
        data: pd.DataFrame,
        continuous_columns: List[str]
    ) -> pd.DataFrame:
        """
        Предобработка данных: кодирование категориальных и дискретизация непрерывных признаков.
        
        Args:
            data (pd.DataFrame): Исходные данные
            continuous_columns (List[str]): Список непрерывных признаков
            
        Returns:
            pd.DataFrame: Предобработанные данные
        """
        processed_data = data.copy()
        
        # Кодирование категориальных признаков
        categorical_columns = processed_data.select_dtypes(
            include=['object', 'category']
        ).columns
        
        for column in categorical_columns:
            if column not in self.label_encoders:
                self.label_encoders[column] = LabelEncoder()
                processed_data[column] = self.label_encoders[column].fit_transform(
                    processed_data[column].astype(str)
                )
            else:
                processed_data[column] = self.label_encoders[column].transform(
                    processed_data[column].astype(str)
                )
        
        # Дискретизация непрерывных признаков
        if continuous_columns:
            if self.discretizer is None:
                self.discretizer = KBinsDiscretizer(
                    n_bins=self.n_bins,
                    encode='ordinal',
                    strategy='uniform'
                )
                processed_data[continuous_columns] = self.discretizer.fit_transform(
                    processed_data[continuous_columns]
                )
            else:
                processed_data[continuous_columns] = self.discretizer.transform(
                    processed_data[continuous_columns]
                )
        
        # Убеждаемся, что все значения целочисленные
        for column in processed_data.columns:
            processed_data[column] = processed_data[column].astype(int)
        
        return processed_data

    def _private_contingency_table(
        self,
        data: pd.DataFrame,
        variables: List[str]
    ) -> np.ndarray:
        """
        Создание приватной таблицы сопряженности.
        
        Args:
            data (pd.DataFrame): Данные
            variables (List[str]): Список переменных
            
        Returns:
            np.ndarray: Приватная таблица сопряженности
        """
        # Создаем таблицу сопряженности
        contingency = pd.crosstab(
            [data[v] for v in variables[:-1]],
            data[variables[-1]]
        ).values
        
        # Добавляем шум Лапласа
        private_contingency = self._add_laplace_noise(contingency)
        
        return private_contingency

    def _create_private_cpd(
        self,
        data: pd.DataFrame,
        node: str,
        parents: List[str]
    ) -> TabularCPD:
        """
        Создание приватной таблицы условных вероятностей.
        
        Args:
            data (pd.DataFrame): Данные
            node (str): Целевая переменная
            parents (List[str]): Список родительских переменных
            
        Returns:
            TabularCPD: Таблица условных вероятностей
        """
        # Получаем количество уникальных значений для целевой переменной
        variable_values = sorted(data[node].unique())
        variable_card = len(variable_values)
        
        if not parents:
            # Если нет родительских переменных, создаем простое распределение
            counts = np.zeros(variable_card)
            for idx, val in enumerate(variable_values):
                counts[idx] = np.sum(data[node] == val)
            
            # Добавляем шум и нормализуем
            counts = self._add_laplace_noise(counts)
            probabilities = (counts / (counts.sum() + 1e-10)).reshape(-1, 1)
            
            return TabularCPD(
                variable=node,
                variable_card=int(variable_card),
                values=probabilities,
                evidence=None,
                evidence_card=None
            )
        
        # Если есть родительские переменные
        evidence_card = []
        for parent in parents:
            parent_values = sorted(data[parent].unique())
            evidence_card.append(len(parent_values))
        
        # Создаем все возможные комбинации значений родительских переменных
        parent_values = [range(card) for card in evidence_card]
        combinations = np.array(np.meshgrid(*parent_values)).T.reshape(-1, len(parents))
        
        # Инициализируем массив для вероятностей
        total_combinations = np.prod(evidence_card)
        probabilities = np.zeros((variable_card, int(total_combinations)))
        
        # Заполняем вероятности для существующих комбинаций
        for i, comb in enumerate(combinations):
            mask = np.ones(len(data), dtype=bool)
            for parent, value in zip(parents, comb):
                mask &= (data[parent] == value)
            
            if mask.any():
                # Получаем значения для текущей комбинации
                current_values = data.loc[mask, node].values
                counts = np.zeros(variable_card)
                
                # Подсчитываем частоты для каждого значения
                for idx, val in enumerate(variable_values):
                    counts[idx] = np.sum(current_values == val)
                
                # Добавляем шум и нормализуем
                counts = self._add_laplace_noise(counts)
                probabilities[:, i] = counts / (counts.sum() + 1e-10)
            else:
                # Если комбинация не встречается, используем равномерное распределение
                probabilities[:, i] = 1.0 / variable_card
        
        # Проверяем, что размерности корректны
        expected_shape = (variable_card, int(np.prod(evidence_card)))
        assert probabilities.shape == expected_shape, \
            f"Expected shape {expected_shape}, got {probabilities.shape}"
        
        return TabularCPD(
            variable=node,
            variable_card=int(variable_card),
            values=probabilities,
            evidence=parents,
            evidence_card=evidence_card
        )
    def fit(
        self,
        data: pd.DataFrame,
        continuous_columns: List[str] = None
    ) -> None:
        """
        Обучение модели на данных.
        
        Args:
            data (pd.DataFrame): Исходные данные
            continuous_columns (List[str]): Список непрерывных признаков
        """
        # Предобработка данных
        processed_data = self._preprocess_data(data, continuous_columns)
        num_tuples, num_attributes = processed_data.shape
        
        # Автоматический подбор k, если не задан
        if self.k is None:
            self.k = self._calculate_k(num_attributes, num_tuples)
            print(f"Автоматически выбран k={self.k}")
        
        # Поиск структуры Байесовской сети с ограничением на число родителей
        hc = HillClimbSearch(processed_data)
        best_model = hc.estimate(
            scoring_method=BDsScore(processed_data),
            max_indegree=self.k  # Ограничение на число родителей
        )
        
        # Создание и обучение модели с приватными CPD
        self.model = BayesianNetwork(best_model.edges())
        
        # Для каждой вершины создаем приватную CPD
        for node in self.model.nodes():
            parents = list(self.model.get_parents(node))
            cpd = self._create_private_cpd(processed_data, node, parents)
            self.model.add_cpds(cpd)

    def generate(self, n_samples: int) -> pd.DataFrame:
        """
        Генерация синтетических данных.
        
        Args:
            n_samples (int): Количество генерируемых образцов
            
        Returns:
            pd.DataFrame: Сгенерированные данные
        """
        if not self.model:
            raise ValueError("Модель не обучена. Сначала выполните метод fit().")
        
        # Генерация синтетических данных
        synthetic_data = self.model.simulate(n_samples=n_samples)
        
        # Обратное преобразование категориальных признаков
        for column, le in self.label_encoders.items():
            synthetic_data[column] = le.inverse_transform(synthetic_data[column])
        
        return synthetic_data
    
import warnings
warnings.simplefilter('ignore')

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

# Инициализация синтезатора с дифференциальной приватностью
private_synthesizer = PrivateBayesianSynthesizer(
    epsilon=0.2,  # параметр приватности
    n_bins=2,     # количество интервалов для дискретизации
    random_state=11
)

# Определение непрерывных признаков
continuous_columns = data.select_dtypes(include=['float64', 'int64']).columns.tolist()

# Обучение модели
private_synthesizer.fit(data, continuous_columns=continuous_columns)

# Генерация синтетических данных
synthetic_samples_df = private_synthesizer.generate(n_samples=data.shape[0])

# Сохранение синтетических данных
synthetic_samples_df.to_csv('./synth_data/privateyyyyyyy_synthetic_data_BN.csv', index=False)
