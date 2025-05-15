import pandas as pd
import numpy as np
import math

import seaborn as sns
import matplotlib.pyplot as plt

from pgmpy.estimators import ParameterEstimator

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

class DPMaximumLikelihoodEstimator(ParameterEstimator):
    def __init__(self, model, data, epsilon=1.0):
        super().__init__(model, data)
        self.epsilon = epsilon  # Параметр приватности

    def get_parameters(self):
        params = []
        for node in self.model.nodes():
            cpd = self.estimate_cpd(node)
            # Добавляем шум Лапласа к counts
            noisy_counts = cpd.values + np.random.laplace(0, 1/self.epsilon, size=cpd.values.shape)
            noisy_cpd = cpd.copy()
            noisy_cpd.values = noisy_counts
            params.append(noisy_cpd)
        return params

def SyntheticBN(real_data):
    label_encoders = {}
    real_data_enc = real_data.copy()

    for column in real_data_enc.select_dtypes(include=['object', 'category']).columns:
        le = LabelEncoder()
        real_data_enc[column] = le.fit_transform(real_data_enc[column])
        label_encoders[column] = le

    real_data_enc = real_data_enc.dropna()
    print(real_data_enc)

    cleaned_real_data = remove_outliers(real_data_enc, exclude_l=['target', 'Card_Category'])

    print(f"Размер данных до очистки: {real_data_enc.shape}")
    print(f"Размер данных после очистки: {cleaned_real_data.shape}")

    # Выбираем непрерывные переменные
    continuous_columns = cleaned_real_data.select_dtypes(include=['float64', 'int64']).columns
    # Дискретизация с использованием равных интервалов (по умолчанию — 5 интервалов)
    discretizer = KBinsDiscretizer(n_bins=5, encode='ordinal', strategy='uniform')
    # Применяем дискретизацию к непрерывным переменным
    cleaned_real_data[continuous_columns] = discretizer.fit_transform(cleaned_real_data[continuous_columns])

    hc = HillClimbSearch(cleaned_real_data)
    # best_model = hc.estimate(scoring_method=BicScore(real_data))
    best_model = hc.estimate(scoring_method=BDsScore(cleaned_real_data)) #  K2Score

    model = BayesianNetwork(best_model.edges())
    # model.fit(real_data_enc, estimator=MaximumLikelihoodEstimator)
    model.fit(real_data_enc, estimator=DPMaximumLikelihoodEstimator)

    #Генерация новых синтетических данных
    sampler = BayesianModelSampling(model)

    synthetic_samples = sampler.forward_sample(size=real_data_enc.shape[0])

    synthetic_samples_df = pd.DataFrame(synthetic_samples)

    synthetic_samples_df

    cleared_synthetic_samples_df = remove_outliers(synthetic_samples_df)
    cleared_synthetic_samples_df.shape

    # Декодируем обратно категориальные переменные
    for column, le in label_encoders.items():
        synthetic_samples_df[column] = le.inverse_transform(synthetic_samples_df[column])

    synthetic_samples_df

    synthetic_samples_df.to_csv('./synth_data/synthetic_data_BN_cls.csv', index=False)


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

SyntheticBN(real_data=real_data)
