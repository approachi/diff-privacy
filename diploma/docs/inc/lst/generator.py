def generate_dataset(self, n, description_file, seed=0):
    set_random_seed(seed)
    self.n = n
    self.description = read_json_file(description_file)

    all_attributes = self.description['meta']['all_attributes']
    candidate_keys = set(self.description['meta']['candidate_keys'])
    self.encoded_dataset = DataGenerator.generate_encoded_dataset(self.n, self.description)
    self.synthetic_dataset = DataFrame(columns=all_attributes)
    for attr in all_attributes:
        attr_info = self.description['attribute_description'][attr]
        column = parse_json(attr_info)

        if attr in self.encoded_dataset:
            self.synthetic_dataset[attr] = column.sample_values_from_binning_indices(self.encoded_dataset[attr])
        elif attr in candidate_keys:
            self.synthetic_dataset[attr] = column.generate_values_as_candidate_key(n)
        else:
            binning_indices = column.sample_binning_indices_in_independent_attribute_mode(n)
            self.synthetic_dataset[attr] = column.sample_values_from_binning_indices(binning_indices)
