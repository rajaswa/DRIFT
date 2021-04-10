from itertools import islice
import matplotlib.pyplot as plt
import nltk
import numpy as np
from nltk import ngrams
import operator

from src.analysis.utils.statistical_measures import find_freq, find_norm_freq


def freq_top_k(text, top_k=200, n=1):
	gram_count_mapping = find_freq(text=text,n=n)
	sorted_gram_count_tuple = sorted(gram_count_mapping.items(), key=operator.itemgetter(1), reverse=True)
	sorted_gram_count_mapping = {" ".join(k):v for k,v in sorted_gram_count_tuple}
	
	if top_k < len(sorted_gram_count_tuple):
		sorted_gram_count_mapping = dict(itertools.islice(sorted_gram_count_mapping.items(),top_k))
	
	return sorted_gram_count_mapping

def norm_freq_top_k(text, top_k=200, n=1):
	gram_count_mapping = find_norm_freq(text=text, n=n)
	sorted_gram_count_tuple = sorted(gram_count_mapping.items(), key=operator.itemgetter(1), reverse=True)
	sorted_gram_count_mapping = {" ".join(k):v for k,v in sorted_gram_count_tuple}
	
	if top_k < len(sorted_gram_count_tuple):
		sorted_gram_count_mapping = dict(itertools.islice(sorted_gram_count_mapping.items(),top_k))

	return sorted_gram_count_mapping










