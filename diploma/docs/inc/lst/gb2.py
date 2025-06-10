                   idx = np.random.choice(list(range(len(mutual_info_list))), p=sampling_distribution)
	else:
            idx = mutual_info_list.index(max(mutual_info_list))

        N.append(parents_pair_list[idx])
        adding_attribute = parents_pair_list[idx][0]
        V.append(adding_attribute)
        rest_attributes.remove(adding_attribute)
        print(f'Adding attribute {adding_attribute}')

    return N