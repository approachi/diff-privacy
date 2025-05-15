import random
import warnings
from itertools import combinations, product, islice, chain
from math import log, ceil
from multiprocessing.pool import Pool

import numpy as np
import pandas as pd
from pandas import DataFrame, merge
from scipy.optimize import fsolve
from sklearn.model_selection import train_test_split
# from DataSynthesizer.lib.utils import mutual_information, normalize_given_distribution, set_random_seed

import json
import random
from string import ascii_lowercase

import numpy as np
from pandas import Series, DataFrame
from sklearn.metrics import mutual_info_score, normalized_mutual_info_score


def set_random_seed(seed=0):
    random.seed(seed)
    np.random.seed(seed)


def mutual_information(labels_x: Series, labels_y: DataFrame):
    """Mutual information of distributions in format of Series or DataFrame.

    Parameters
    ----------
    labels_x : Series
    labels_y : DataFrame
    """
    if labels_y.shape[1] == 1:
        labels_y = labels_y.iloc[:, 0]
    else:
        labels_y = labels_y.apply(lambda x: ' '.join(x.values), axis=1)

    return mutual_info_score(labels_x, labels_y)


def pairwise_attributes_mutual_information(dataset):
    """Compute normalized mutual information for all pairwise attributes. Return a DataFrame."""
    sorted_columns = sorted(dataset.columns)
    mi_df = DataFrame(columns=sorted_columns, index=sorted_columns, dtype=float)
    for row in mi_df.columns:
        for col in mi_df.columns:
            mi_df.loc[row, col] = normalized_mutual_info_score(dataset[row].astype(str),
                                                               dataset[col].astype(str),
                                                               average_method='arithmetic')
    return mi_df


def normalize_given_distribution(frequencies):
    distribution = np.array(frequencies, dtype=float)
    distribution = distribution.clip(0)  # replace negative values with 0
    summation = distribution.sum()
    if summation > 0:
        if np.isinf(summation):
            return normalize_given_distribution(np.isinf(distribution))
        else:
            return distribution / summation
    else:
        return np.full_like(distribution, 1 / distribution.size)


def read_json_file(json_file):
    with open(json_file, 'r') as file:
        return json.load(file)


def infer_numerical_attributes_in_dataframe(dataframe):
    describe = dataframe.describe()
    # DataFrame.describe() usually returns 8 rows.
    if describe.shape[0] == 8:
        return set(describe.columns)
    # DataFrame.describe() returns less than 8 rows when there is no numerical attribute.
    else:
        return set()


def display_bayesian_network(bn):
    length = 0
    for child, _ in bn:
        if len(child) > length:
            length = len(child)

    print('Constructed Bayesian network:')
    for child, parents in bn:
        print("    {0:{width}} has parents {1}.".format(child, parents, width=length))


def generate_random_string(length):
    return ''.join(np.random.choice(list(ascii_lowercase), size=length))

"""
This module is based on PrivBayes in the following paper:

Zhang J, Cormode G, Procopiuc CM, Srivastava D, Xiao X.
PrivBayes: Private Data Release via Bayesian Networks.
"""



def calculate_sensitivity(num_tuples, child, parents, attr_to_is_binary):
    """Sensitivity function for Bayesian network construction. PrivBayes Lemma 1.

    Parameters
    ----------
    num_tuples : int
        Number of tuples in sensitive dataset.

    Return
    --------
    int
        Sensitivity value.
    """
    if attr_to_is_binary[child] or (len(parents) == 1 and attr_to_is_binary[parents[0]]):
        a = log(num_tuples) / num_tuples
        b = (num_tuples - 1) / num_tuples
        b_inv = num_tuples / (num_tuples - 1)
        return a + b * log(b_inv)
    else:
        a = (2 / num_tuples) * log((num_tuples + 1) / 2)
        b = (1 - 1 / num_tuples) * log(1 + 2 / (num_tuples - 1))
        return a + b


def calculate_delta(num_attributes, sensitivity, epsilon):
    """Computing delta, which is a factor when applying differential privacy.

    More info is in PrivBayes Section 4.2 "A First-Cut Solution".

    Parameters
    ----------
    num_attributes : int
        Number of attributes in dataset.
    sensitivity : float
        Sensitivity of removing one tuple.
    epsilon : float
        Parameter of differential privacy.
    """
    return (num_attributes - 1) * sensitivity / epsilon


def usefulness_minus_target(k, num_attributes, num_tuples, target_usefulness=5, epsilon=0.1):
    """Usefulness function in PrivBayes.

    Parameters
    ----------
    k : int
        Max number of degree in Bayesian networks construction
    num_attributes : int
        Number of attributes in dataset.
    num_tuples : int
        Number of tuples in dataset.
    target_usefulness : int or float
    epsilon : float
        Parameter of differential privacy.
    """
    if k == num_attributes:
        print('here')
        usefulness = target_usefulness
    else:
        usefulness = num_tuples * epsilon / ((num_attributes - k) * (2 ** (k + 3)))  # PrivBayes Lemma 3
    return usefulness - target_usefulness


def calculate_k(num_attributes, num_tuples, target_usefulness=4, epsilon=0.1):
    """Calculate the maximum degree when constructing Bayesian networks. See PrivBayes Lemma 3."""
    default_k = 3
    initial_usefulness = usefulness_minus_target(default_k, num_attributes, num_tuples, 0, epsilon)
    if initial_usefulness > target_usefulness:
        return default_k
    else:
        arguments = (num_attributes, num_tuples, target_usefulness, epsilon)
        warnings.filterwarnings("error")
        try:
            ans = fsolve(usefulness_minus_target, np.array([int(num_attributes / 2)]), args=arguments)[0]
            ans = ceil(ans)
        except RuntimeWarning:
            print("Warning: k is not properly computed!")
            ans = default_k
        if ans < 1 or ans > num_attributes:
            ans = default_k
        return ans


def worker(paras):
    child, V, num_parents, split, dataset = paras
    parents_pair_list = []
    mutual_info_list = []

    if split + num_parents - 1 < len(V):
        for other_parents in combinations(V[split + 1:], num_parents - 1):
            parents = list(other_parents)
            parents.append(V[split])
            parents_pair_list.append((child, parents))
            # TODO consider to change the computation of MI by combined integers instead of strings.
            mi = mutual_information(dataset[child], dataset[parents])
            mutual_info_list.append(mi)

    return parents_pair_list, mutual_info_list


def greedy_bayes(dataset: DataFrame, k: int, epsilon: float, seed=0):
    """Construct a Bayesian Network (BN) using greedy algorithm.

    Parameters
    ----------
    dataset : DataFrame
        Input dataset, which only contains categorical attributes.
    k : int
        Maximum degree of the constructed BN. If k=0, k is automatically calculated.
    epsilon : float
        Parameter of differential privacy.
    seed : int or float
        Seed for the randomness in BN generation.
    """
    set_random_seed(seed)
    dataset: DataFrame = dataset.astype(str, copy=False)
    num_tuples, num_attributes = dataset.shape
    if not k:
        k = calculate_k(num_attributes, num_tuples)

    attr_to_is_binary = {attr: dataset[attr].unique().size <= 2 for attr in dataset}

    print('================ Constructing Bayesian Network (BN) ================')
    root_attribute = random.choice(dataset.columns)
    V = [root_attribute]
    rest_attributes = list(dataset.columns)
    rest_attributes.remove(root_attribute)
    print(f'Adding ROOT {root_attribute}')
    N = []
    while rest_attributes:
        parents_pair_list = []
        mutual_info_list = []

        num_parents = min(len(V), k)
        tasks = [(child, V, num_parents, split, dataset) for child, split in
                 product(rest_attributes, range(len(V) - num_parents + 1))]
        with Pool() as pool:
            res_list = pool.map(worker, tasks)

        for res in res_list:
            parents_pair_list += res[0]
            mutual_info_list += res[1]

        if epsilon:
            sampling_distribution = exponential_mechanism(epsilon, mutual_info_list, parents_pair_list, attr_to_is_binary,
                                                          num_tuples, num_attributes)
            idx = np.random.choice(list(range(len(mutual_info_list))), p=sampling_distribution)
        else:
            idx = mutual_info_list.index(max(mutual_info_list))

        N.append(parents_pair_list[idx])
        adding_attribute = parents_pair_list[idx][0]
        V.append(adding_attribute)
        rest_attributes.remove(adding_attribute)
        print(f'Adding attribute {adding_attribute}')

    print('========================== BN constructed ==========================')

    return N


def exponential_mechanism(epsilon, mutual_info_list, parents_pair_list, attr_to_is_binary, num_tuples, num_attributes):
    """Applied in Exponential Mechanism to sample outcomes."""
    delta_array = []
    for (child, parents) in parents_pair_list:
        sensitivity = calculate_sensitivity(num_tuples, child, parents, attr_to_is_binary)
        delta = calculate_delta(num_attributes, sensitivity, epsilon)
        delta_array.append(delta)

    mi_array = np.array(mutual_info_list) / (2 * np.array(delta_array))
    mi_array = np.exp(mi_array)
    mi_array = normalize_given_distribution(mi_array)
    return mi_array


def laplace_noise_parameter(k, num_attributes, num_tuples, epsilon):
    """The noises injected into conditional distributions.

    Note that these noises are over counts, instead of the probability distributions in PrivBayes Algorithm 1.
    """
    return (num_attributes - k) / epsilon


def get_noisy_distribution_of_attributes(attributes, encoded_dataset, epsilon=0.1):
    data = encoded_dataset.copy().loc[:, attributes]
    data['count'] = 1
    stats = data.groupby(attributes).sum()

    iterables = [range(int(encoded_dataset[attr].max()) + 1) for attr in attributes]
    products = product(*iterables)

    def grouper_it(iterable, n):
        while True:
            chunk_it = islice(iterable, n)
            try:
                first_el = next(chunk_it)
            except StopIteration:
                return
            yield chain((first_el,), chunk_it)

    full_space = None
    for item in grouper_it(products, 1000000):
        if full_space is None:
            full_space = DataFrame(columns=attributes, data=list(item))
        else:
            data_frame_append = DataFrame(columns=attributes, data=list(item))
            full_space = pd.concat([full_space, data_frame_append], ignore_index=True)

    stats.reset_index(inplace=True)
    stats = merge(full_space, stats, how='left')
    stats.fillna(0, inplace=True)

    if epsilon:
        k = len(attributes) - 1
        num_tuples, num_attributes = encoded_dataset.shape
        noise_para = laplace_noise_parameter(k, num_attributes, num_tuples, epsilon)
        laplace_noises = np.random.laplace(0, scale=noise_para, size=stats.index.size)
        stats['count'] += laplace_noises
        stats.loc[stats['count'] < 0, 'count'] = 0

    return stats


def construct_noisy_conditional_distributions(bayesian_network, encoded_dataset, epsilon=0.1):
    """See more in Algorithm 1 in PrivBayes."""

    print('construct_noisy_conditional_distributions start')

    k = len(bayesian_network[-1][1])
    conditional_distributions = {}

    # first k+1 attributes
    root = bayesian_network[0][1][0]
    kplus1_attributes = [root]
    for child, _ in bayesian_network[:k]:
        kplus1_attributes.append(child)

    noisy_dist_of_kplus1_attributes = get_noisy_distribution_of_attributes(kplus1_attributes, encoded_dataset, epsilon)

    # generate noisy distribution of root attribute.
    root_stats = noisy_dist_of_kplus1_attributes.loc[:, [root, 'count']].groupby(root).sum()['count']
    conditional_distributions[root] = normalize_given_distribution(root_stats).tolist()

    for idx, (child, parents) in enumerate(bayesian_network):
        conditional_distributions[child] = {}

        if idx <= k - 2:
            stats = noisy_dist_of_kplus1_attributes.copy().loc[:, parents + [child, 'count']]
            stats = stats.groupby(parents + [child], as_index=False).sum()
        elif idx == k - 1:
            stats = noisy_dist_of_kplus1_attributes.loc[:, parents + [child, 'count']]
        else:
            stats = get_noisy_distribution_of_attributes(parents + [child], encoded_dataset, epsilon)
            stats = stats.loc[:, parents + [child, 'count']]

        parents_grouper = parents[0] if len(parents) == 1 else parents
        for parents_instance, stats_sub in stats.groupby(parents_grouper):
            stats_sub = stats_sub.sort_values(by=child)
            dist = normalize_given_distribution(stats_sub['count']).tolist()

            parents_key = str([parents_instance]) if len(parents) == 1 else str(list(parents_instance))
            conditional_distributions[child][parents_key] = dist

    print('construct_noisy_conditional_distributions end')
    return conditional_distributions

def generate_synthetic_data(bayesian_network, conditional_distributions, num_samples, seed=None):
    """
    Генерация синтетических данных на основе Байесовской сети и условных распределений.
    
    Parameters
    ----------
    bayesian_network : list of tuples
        Структура Байесовской сети в виде списка кортежей (child, parents).
    conditional_distributions : dict
        Словарь условных распределений вероятностей.
    num_samples : int
        Количество генерируемых образцов.
    seed : int, optional
        Seed для генератора случайных чисел.
        
    Returns
    -------
    pandas.DataFrame
        Датафрейм с синтетическими данными.
    """
    print('generate_synthetic_data start')
    if seed is not None:
        np.random.seed(seed)
    
    # Определяем порядок генерации (топологическая сортировка)
    nodes_order = []
    visited = set()
    
    def topological_sort(node, parents):
        if node in visited:
            return
        visited.add(node)
        for parent in parents:
            if parent not in visited:
                # Ищем родителей этого родителя
                parent_idx = next((i for i, (n, p) in enumerate(bayesian_network) if n == parent), -1)
                if parent_idx >= 0:
                    topological_sort(parent, bayesian_network[parent_idx][1])
        nodes_order.append(node)
    
    # Добавляем корневой атрибут
    root = bayesian_network[0][1][0] if bayesian_network[0][1] else bayesian_network[0][0]
    if not bayesian_network[0][1]:  # Если у корня нет родителей
        nodes_order.append(root)
    else:
        topological_sort(root, [])
    
    # Добавляем остальные узлы в топологическом порядке
    for child, parents in bayesian_network:
        if child not in visited:
            topological_sort(child, parents)
    
    # Генерация данных
    synthetic_data = {}
    
    for node in nodes_order:
        # Находим родителей для текущего узла
        parents = next((p for n, p in bayesian_network if n == node), [])
        
        if not parents:
            # Для корневого узла используем безусловное распределение
            probs = conditional_distributions[node]
            values = np.random.choice(
                range(len(probs)),
                size=num_samples,
                p=probs
            )
        else:
            values = np.zeros(num_samples, dtype=int)
            
            # Для каждого образца находим соответствующее условное распределение
            for i in range(num_samples):
                parent_values = [synthetic_data[parent][i] for parent in parents]
                parent_key = str([parent_values[0]]) if len(parents) == 1 else str(parent_values)
                
                if parent_key in conditional_distributions[node]:
                    probs = conditional_distributions[node][parent_key]
                else:
                    # Если комбинация не встречается, используем равномерное распределение
                    domain_size = max(len(dist) for dist in conditional_distributions[node].values())
                    probs = [1.0 / domain_size] * domain_size
                
                values[i] = np.random.choice(range(len(probs)), p=probs)
        
        synthetic_data[node] = values
    
    print('generate_synthetic_data end')
    # Преобразуем в датафрейм
    return pd.DataFrame(synthetic_data)

if __name__ == '__main__':

    data = pd.read_csv('./data/BankChurners.csv')
    churn_map = {'Existing Customer': 0,
            'Attrited Customer': 1}

    data['target'] = data['Attrition_Flag'].map(churn_map)
    data.drop(['CLIENTNUM',
        'Naive_Bayes_Classifier_Attrition_Flag_Card_Category_Contacts_Count_12_mon_Dependent_count_Education_Level_Months_Inactive_12_mon_1',
     'Naive_Bayes_Classifier_Attrition_Flag_Card_Category_Contacts_Count_12_mon_Dependent_count_Education_Level_Months_Inactive_12_mon_2'],
        axis=1, inplace=True)

    data[data.columns[15:16]] = data[data.columns[15:16]].astype(float)

    data.drop('Attrition_Flag', axis=1, inplace=True)
    data = data[['Gender', 'Dependent_count', 'Marital_Status', 'Income_Category', 'Card_Category', 
             'Total_Relationship_Count', 'Credit_Limit', 'Avg_Open_To_Buy', 'Total_Trans_Amt', 
             'Total_Trans_Ct', 'Customer_Age', 'Months_on_book', 
             'Total_Amt_Chng_Q4_Q1', 'target']]
    
    X_train, X_test, y_train, y_test = train_test_split(data.drop(['target'], axis=1),
                                                    data['target'], test_size=0.1,
                                                    shuffle=True, stratify=data['target'],
                                                    random_state=42)
    
    real_data = X_train.copy()
    real_data['target'] = y_train
    categorical_features = list(real_data.loc[:, real_data.dtypes == object].columns)

    print(categorical_features)


    bn = greedy_bayes(real_data, k=3, epsilon=0.1)
    distributions = construct_noisy_conditional_distributions(bn, real_data, epsilon=0.1)
    synthetic_data = generate_synthetic_data(bn, distributions, num_samples=1000)
    synthetic_data.to_csv('./synth_data/pr11111111_synthetic_data_BN.csv', index=False)