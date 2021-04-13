import argparse
import os
import json
from src.analysis import *
import glob
from tqdm.auto import tqdm

parser = argparse.ArgumentParser()
parser.add_argument("--folder", type=str, default="data/files")
parser.add_argument("--keyword_method", type=str, choices=["freq", "norm_freq"], default="norm_freq")
parser.add_argument("--top_k", type=int, default=200)
parser.add_argument("--truncate_popular", type=int, default=200)
args = parser.parse_args()

FOLDER = args.folder
KEYWORD_METHOD = args.keyword_method
TOP_K = args.top_k
TRUNCATE_POPULAR = args.truncate_popular


compass_file_path = os.path.join(FOLDER, "compass.txt")
with open(compass_file_path, "r") as compass_f:
	compass_text = compass_f.read()

if KEYWORD_METHOD == "freq":
	compass_unigrams = freq_top_k(compass_text, top_k=TOP_K, n=1)
	compass_bigrams = freq_top_k(compass_text, top_k=TOP_K, n=2)
	compass_trigrams = freq_top_k(compass_text, top_k=TOP_K, n=3)
elif KEYWORD_METHOD == "norm_freq":
	compass_unigrams = norm_freq_top_k(compass_text, top_k=TOP_K, n=1)
	compass_bigrams = norm_freq_top_k(compass_text, top_k=TOP_K, n=2)
	compass_trigrams = norm_freq_top_k(compass_text, top_k=TOP_K, n=3)

text_file_paths = glob.glob(os.path.join(FOLDER, "*"))
text_file_paths = [text_file_path for text_file_path in text_file_paths if "compass" not in text_file_path]
text_file_paths.sort()
print(text_file_paths)

year_wise_unigram_count_mappings = {}
year_wise_bigram_count_mappings = {}
year_wise_trigram_count_mappings = {}
year_wise_word_prod_mappings = {}

text_file_paths_tqdm = tqdm(text_file_paths)
for text_file_path in text_file_paths_tqdm:
	year = os.path.split(text_file_path)[-1].split(".")[0]
	text_file_paths_tqdm.set_description_str(year)
	if not os.path.exists(f"outputs/keyword_ext/{year}"):
		os.makedirs(f"outputs/keyword_ext/{year}")
	if not os.path.exists(f"outputs/wordcloud"):
		os.makedirs(f"outputs/wordcloud")
	with open(text_file_path, "r") as f:
		text = f.read()
	
	if KEYWORD_METHOD == "freq":
		unigrams = freq_top_k(text, top_k=TOP_K, n=1)
		bigrams = freq_top_k(text, top_k=TOP_K, n=2)
		trigrams = freq_top_k(text, top_k=TOP_K, n=3)
		year_wise_unigram_count_mappings[year] = find_freq(text, n=1)
		year_wise_bigram_count_mappings[year] = find_freq(text, n=2)
		year_wise_trigram_count_mappings[year] = find_freq(text, n=3)
	elif KEYWORD_METHOD == "norm_freq":
		unigrams = norm_freq_top_k(text, top_k=TOP_K, n=1)
		bigrams = norm_freq_top_k(text, top_k=TOP_K, n=2)
		trigrams = norm_freq_top_k(text, top_k=TOP_K, n=3)
		year_wise_unigram_count_mappings[year] = find_norm_freq(text, n=1)
		year_wise_bigram_count_mappings[year] = find_norm_freq(text, n=2)
		year_wise_trigram_count_mappings[year] = find_norm_freq(text, n=3)

	with open(f"outputs/keyword_ext/{year}/unigrams.json", "w") as f:
		json.dump(unigrams, f)
	with open(f"outputs/keyword_ext/{year}/bigrams.json", "w") as f:
		json.dump(bigrams, f)
	with open(f"outputs/keyword_ext/{year}/trigrams.json", "w") as f:
		json.dump(trigrams, f)

	make_word_cloud(corpus=text, save_path=f"outputs/wordcloud/{year}.svg", truncate_popular=TRUNCATE_POPULAR)

	year_wise_word_prod_mappings[year] = find_productivity(list(compass_unigrams.keys()), text, 2)

compass_unigrams_tqdm = tqdm(compass_unigrams)
compass_unigrams_tqdm.set_description_str("Unigram Freq. Plots")
for word in compass_unigrams_tqdm:
	save_path = f"outputs/freq_plots/unigrams/"
	prod_save_path = f"outputs/prod_plots/unigrams/"
	if not os.path.exists(save_path):
		os.makedirs(save_path)
	if not os.path.exists(prod_save_path):
		os.makedirs(prod_save_path)
	save_path = os.path.join(save_path,word+".svg")
	prod_save_path = os.path.join(prod_save_path,word+".svg")
	plot_freq(word, year_wise_unigram_count_mappings, save_path)
	plot_prod(word, year_wise_word_prod_mappings, prod_save_path)

compass_bigrams_tqdm = tqdm(compass_bigrams)
compass_bigrams_tqdm.set_description_str("Bigram Freq. Plots")
for word in compass_bigrams_tqdm:
	save_path = f"outputs/freq_plots/bigrams/"
	if not os.path.exists(save_path):
		os.makedirs(save_path)
	save_path = os.path.join(save_path,word+".svg")
	plot_freq(word, year_wise_bigram_count_mappings, save_path)

compass_trigrams_tqdm = tqdm(compass_trigrams)
compass_trigrams_tqdm.set_description_str("Trigram Freq. Plots")
for word in compass_trigrams_tqdm:
	save_path = f"outputs/freq_plots/trigrams/"
	if not os.path.exists(save_path):
		os.makedirs(save_path)
	save_path = os.path.join(save_path,word+".svg")
	plot_freq(word, year_wise_trigram_count_mappings, save_path)
