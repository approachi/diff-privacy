
import numpy as np
import pandas as pd
from sklearn.base import BaseEstimator
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder, StandardScaler
from sklearn.metrics import (
    accuracy_score, precision_score, recall_score, 
    f1_score, roc_auc_score, mean_squared_error
)
from diffprivlib.models import (
    LogisticRegression as PrivateLogisticRegression,
    GaussianNB as PrivateGaussianNB,
    RandomForestClassifier as PrivateRandomForestClassifier,
    LinearRegression as PrivateLinearRegression
)
from typing import Dict, Tuple, Union, List

class DifferentialPrivacyML:
    """
    Класс для обучения моделей с дифференциальной приватностью на синтетических данных.
    """
    
    def __init__(
        self, 
        model_type: str, 
        epsilon: float = 1.0,
        task_type: str = 'classification',
        random_state: int = 42
    ):
        """
        Инициализация класса.
        
        Args:
            model_type (str): Тип модели ('lr', 'rf', 'nb', 'linreg')
            epsilon (float): Параметр силы приватности
            task_type (str): Тип задачи ('classification' или 'regression')
            random_state (int): Seed для воспроизводимости результатов
        """
        self.model_type = model_type.lower()
        self.epsilon = epsilon
        self.task_type = task_type
        self.random_state = random_state
        self.model = self._initialize_model()
        self.label_encoders = {}
        self.scaler = StandardScaler()
        self.is_fitted = False
        
    def _initialize_model(self) -> BaseEstimator:
        """
        Инициализация модели с дифференциальной приватностью.
        
        Returns:
            BaseEstimator: Инициализированная модель
        """
        if self.task_type == 'classification':
            if self.model_type == 'lr':
                return PrivateLogisticRegression(epsilon=self.epsilon, random_state=self.random_state)
            elif self.model_type == 'rf':
                return PrivateRandomForestClassifier(epsilon=self.epsilon, random_state=self.random_state)
            elif self.model_type == 'nb':
                return PrivateGaussianNB(epsilon=self.epsilon)
            else:
                raise ValueError("Для классификации поддерживаются модели: 'lr', 'rf', 'nb'")
        else:
            if self.model_type == 'linreg':
                return PrivateLinearRegression(epsilon=self.epsilon)
            else:
                raise ValueError("Для регрессии поддерживается только 'linreg'")

    def _preprocess_data(
        self, 
        data: pd.DataFrame, 
        categorical_features: List[str] = None,
        fit: bool = True
    ) -> pd.DataFrame:
        """
        Предобработка данных.
        
        Args:
            data (pd.DataFrame): Входные данные
            categorical_features (List[str]): Список категориальных признаков
            fit (bool): Флаг для обучения или применения преобразований
            
        Returns:
            pd.DataFrame: Предобработанные данные
        """
        processed_data = data.copy()
        
        # Обработка категориальных признаков
        if categorical_features:
            for column in categorical_features:
                if fit:
                    self.label_encoders[column] = LabelEncoder()
                    processed_data[column] = self.label_encoders[column].fit_transform(
                        processed_data[column]
                    )
                else:
                    processed_data[column] = self.label_encoders[column].transform(
                        processed_data[column]
                    )
        
        # Масштабирование числовых признаков
        numeric_features = processed_data.select_dtypes(
            include=['int64', 'float64']
        ).columns
        
        if len(numeric_features) > 0:
            if fit:
                processed_data[numeric_features] = self.scaler.fit_transform(
                    processed_data[numeric_features]
                )
            else:
                processed_data[numeric_features] = self.scaler.transform(
                    processed_data[numeric_features]
                )
                
        return processed_data

    def train(
        self, 
        synthetic_data: pd.DataFrame,
        target_column: str,
        categorical_features: List[str] = None,
        test_size: float = 0.2
    ) -> Dict:
        """
        Обучение модели на синтетических данных.
        
        Args:
            synthetic_data (pd.DataFrame): Синтетические данные
            target_column (str): Название целевой переменной
            categorical_features (List[str]): Список категориальных признаков
            test_size (float): Размер тестовой выборки
            
        Returns:
            Dict: Метрики качества модели
        """
        X = synthetic_data.drop(columns=[target_column])
        y = synthetic_data[target_column]
        
        # Разделение на обучающую и тестовую выборки
        X_train, X_test, y_train, y_test = train_test_split(
            X, y, test_size=test_size, random_state=self.random_state,
            stratify=y if self.task_type == 'classification' else None
        )
        
        # Предобработка данных
        X_train_processed = self._preprocess_data(
            X_train, categorical_features, fit=True
        )
        X_test_processed = self._preprocess_data(
            X_test, categorical_features, fit=False
        )
        
        # Обучение модели
        self.model.fit(X_train_processed, y_train)
        self.is_fitted = True
        
        # Получение предсказаний
        if self.task_type == 'classification':
            train_pred_proba = self.model.predict_proba(X_train_processed)[:, 1]
            test_pred_proba = self.model.predict_proba(X_test_processed)[:, 1]
            train_pred = self.model.predict(X_train_processed)
            test_pred = self.model.predict(X_test_processed)
            
            metrics = {
                'train': {
                    'accuracy': accuracy_score(y_train, train_pred),
                    'precision': precision_score(y_train, train_pred),
                    'recall': recall_score(y_train, train_pred),
                    'f1': f1_score(y_train, train_pred),
                    'roc_auc': roc_auc_score(y_train, train_pred_proba)
                },
                'test': {
                    'accuracy': accuracy_score(y_test, test_pred),
                    'precision': precision_score(y_test, test_pred),
                    'recall': recall_score(y_test, test_pred),
                    'f1': f1_score(y_test, test_pred),
                    'roc_auc': roc_auc_score(y_test, test_pred_proba)
                }
            }
        else:
            train_pred = self.model.predict(X_train_processed)
            test_pred = self.model.predict(X_test_processed)
            
            metrics = {
                'train': {
                    'mse': mean_squared_error(y_train, train_pred),
                    'rmse': np.sqrt(mean_squared_error(y_train, train_pred))
                },
                'test': {
                    'mse': mean_squared_error(y_test, test_pred),
                    'rmse': np.sqrt(mean_squared_error(y_test, test_pred))
                }
            }
        
        self.metrics = metrics
        return metrics

    def predict(
        self, 
        X: pd.DataFrame, 
        categorical_features: List[str] = None
    ) -> np.ndarray:
        """
        Предсказание на новых данных.
        
        Args:
            X (pd.DataFrame): Данные для предсказания
            categorical_features (List[str]): Список категориальных признаков
            
        Returns:
            np.ndarray: Предсказанные значения
        """
        if not self.is_fitted:
            raise ValueError("Модель не обучена. Сначала выполните метод train().")
            
        X_processed = self._preprocess_data(
            X, categorical_features, fit=False
        )
        
        if self.task_type == 'classification':
            return self.model.predict_proba(X_processed)[:, 1]
        return self.model.predict(X_processed)

    def get_metrics(self) -> Dict:
        """
        Получение метрик качества модели.
        
        Returns:
            Dict: Словарь с метриками качества
        """
        if not self.is_fitted:
            raise ValueError("Модель не обучена. Сначала выполните метод train().")
        return self.metrics
    

# Пример использования
# Загрузка синтетических данных
synthetic_samples_df = pd.read_csv('./synth_data/synthetic_data_BN_cls.csv')

# Определение категориальных признаков
categorical_features = ['Gender', 'Marital_Status', 'Income_Category', 'Card_Category']

# Инициализация модели с дифференциальной приватностью
dp_model = DifferentialPrivacyML(
    model_type='lr',  # можно использовать 'lr', 'rf', 'nb' для классификации
    epsilon=1.0,      # параметр приватности
    task_type='classification'
)

# Обучение модели на синтетических данных
metrics = dp_model.train(
    synthetic_data=synthetic_samples_df,
    target_column='target',
    categorical_features=categorical_features
)

# Вывод метрик качества
print("\nМетрики на обучающей выборке:")
for metric, value in metrics['train'].items():
    print(f"{metric}: {value:.4f}")

print("\nМетрики на тестовой выборке:")
for metric, value in metrics['test'].items():
    print(f"{metric}: {value:.4f}")

# Получение предсказаний для новых данных
new_data = synthetic_samples_df.drop('target', axis=1).head()  # пример новых данных
predictions = dp_model.predict(new_data, categorical_features=categorical_features)
print("\nПредсказания для новых данных:")
print(predictions)