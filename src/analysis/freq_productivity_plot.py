def plot_freq(word, year_wise_word_count_mappings):
    years = []
    freqs = []
    for year in year_wise_word_count_mappings:
        word_count_mapping = year_wise_word_count_mappings[year]
        if word in word_count_mapping:
            freqs.append(word_count_mapping[word])
        else:
            freqs.append(0)
        years.append(year)
    return years, freqs


def get_freq_info(word, year_wise_prods_mappings):
    years = []
    prods = []
    for year in year_wise_prods_mappings:
        word_prod_mapping = year_wise_prods_mappings[year]
        if word in word_prod_mapping:
            prods.append(word_prod_mapping[word])
        else:
            prods.append(0)
        years.append(year)
    return years, prods
