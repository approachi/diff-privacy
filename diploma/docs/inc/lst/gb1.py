def bayes_struct(dataset: DataFrame, k: int, epsilon: float, seed=0):
    set_random_seed(seed)
    dataset: DataFrame = dataset.astype(str, copy=False)
    num_tuples, num_attributes = dataset.shape
    if not k:
        k = calculate_k(num_attributes, num_tuples)

    attr_to_is_binary = {attr: dataset[attr].unique().size <= 2 for attr in dataset}

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
            sampling_distribution = exponential_mechanism(
		epsilon,
		mutual_info_list,
		parents_pair_list,
		attr_to_is_binary,
                num_tuples,
		num_attributes
		)
