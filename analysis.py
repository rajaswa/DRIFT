import argparse
import os
import json
from src.analysis import *
import glob
from tqdm.auto import tqdm

parser = argparse.ArgumentParser()
parser.add_argument("--folder", type=str, default="./data/files")
parser.add_argument("--cache_folder", type=str, default="./cache")
parser.add_argument("--output_folder", type=str, default="./outputs")
parser.add_argument("--keyword_method", type=str, choices=["freq", "norm_freq"], default="norm_freq")
parser.add_argument("--top_k", type=int, default=200)
parser.add_argument("--truncate_popular", type=int, default=200)
args = parser.parse_args()

FOLDER = args.folder
CACHE_FOLDER = args.cache_folder
OUTPUT_FOLDER = args.output_folder
KEYWORD_METHOD = args.keyword_method
TOP_K = args.top_k
TRUNCATE_POPULAR = args.truncate_popular

if not os.path.exists(CACHE_FOLDER):
	os.makedirs(CACHE_FOLDER)

if not os.path.exists(OUTPUT_FOLDER):
	os.makedirs(OUTPUT_FOLDER)

# READ COMPASS FILE
compass_file_path = os.path.join(FOLDER, "compass.txt")
with open(compass_file_path, "r") as compass_f:
	compass_text = compass_f.read()


# FIND KEYWORDS FROM COMPASS TEXT
COMPASS_KEYWORDS_PATH =  os.path.join(CACHE_FOLDER, "keywords", "compass")
COMPASS_UNIGRAMS_PATH = os.path.join(COMPASS_KEYWORDS_PATH, "unigrams.json")
COMPASS_BIGRAMS_PATH = os.path.join(COMPASS_KEYWORDS_PATH, "bigrams.json")
COMPASS_TRIGRAMS_PATH = os.path.join(COMPASS_KEYWORDS_PATH, "trigrams.json")
if not os.path.exists(COMPASS_KEYWORDS_PATH):
	os.makedirs(COMPASS_KEYWORDS_PATH)

if KEYWORD_METHOD == "freq":
	compass_unigrams = freq_top_k(text=compass_text, save_load_path=COMPASS_UNIGRAMS_PATH, top_k=TOP_K, n=1, overwrite=False)
	compass_bigrams = freq_top_k(text=compass_text, save_load_path=COMPASS_BIGRAMS_PATH, top_k=TOP_K, n=2, overwrite=False)
	compass_trigrams = freq_top_k(text=compass_text, save_load_path=COMPASS_TRIGRAMS_PATH, top_k=TOP_K, n=3, overwrite=False)
elif KEYWORD_METHOD == "norm_freq":
	compass_unigrams = norm_freq_top_k(text=compass_text, save_load_path=COMPASS_UNIGRAMS_PATH, top_k=TOP_K, n=1, overwrite=False)
	compass_bigrams = norm_freq_top_k(text=compass_text, save_load_path=COMPASS_BIGRAMS_PATH, top_k=TOP_K, n=2, overwrite=False)
	compass_trigrams = norm_freq_top_k(text=compass_text, save_load_path=COMPASS_TRIGRAMS_PATH, top_k=TOP_K, n=3, overwrite=False)
print()

text_file_paths = glob.glob(os.path.join(FOLDER, "*"))
text_file_paths = [text_file_path for text_file_path in text_file_paths if "compass" not in text_file_path]
text_file_paths.sort()
print("Text Files Found:", text_file_paths)
print()

year_wise_unigram_mappings = {}
year_wise_bigram_mappings = {}
year_wise_trigram_mappings = {}
year_wise_word_prod_mappings = {}


for text_file_path in text_file_paths:
	
	year = os.path.split(text_file_path)[-1].split(".")[0]
	year_keywords_path = os.path.join(CACHE_FOLDER, "keywords", year)
	year_unigrams_path = os.path.join(year_keywords_path, "unigrams.json")
	year_bigrams_path = os.path.join(year_keywords_path, "bigrams.json")
	year_trigrams_path = os.path.join(year_keywords_path, "trigrams.json")
	if not os.path.exists(year_keywords_path):
		os.makedirs(year_keywords_path)

	with open(text_file_path,"r") as year_text_f:
		year_text = year_text_f.read()
	
	if KEYWORD_METHOD == "freq":
		year_wise_unigram_mappings[year] = freq_top_k(text=year_text, save_load_path=year_unigrams_path, top_k=TOP_K, n=1, overwrite=False)
		year_wise_bigram_mappings[year] = freq_top_k(text=year_text, save_load_path=year_bigrams_path, top_k=TOP_K, n=2, overwrite=False)
		year_wise_trigram_mappings[year] = freq_top_k(text=year_text, save_load_path=year_trigrams_path, top_k=TOP_K, n=3, overwrite=False)
	elif KEYWORD_METHOD == "norm_freq":
		year_wise_unigram_mappings[year] = norm_freq_top_k(text=year_text, save_load_path=year_unigrams_path, top_k=TOP_K, n=1, overwrite=False)
		year_wise_bigram_mappings[year] = norm_freq_top_k(text=year_text, save_load_path=year_bigrams_path, top_k=TOP_K, n=2, overwrite=False)
		year_wise_trigram_mappings[year] = norm_freq_top_k(text=year_text, save_load_path=year_trigrams_path, top_k=TOP_K, n=3, overwrite=False)

	WORD_CLOUD_PATH = os.path.join(OUTPUT_FOLDER, "wordclouds")
	if not os.path.exists(WORD_CLOUD_PATH):
		os.makedirs(WORD_CLOUD_PATH)
	WORD_CLOUD_YEAR_PATH = os.path.join(WORD_CLOUD_PATH, f"{year}.svg")
	print(f"Making the Word Cloud for {year} and saving at {WORD_CLOUD_YEAR_PATH}")
	make_word_cloud(corpus=year_text, save_path=WORD_CLOUD_YEAR_PATH, truncate_popular=TRUNCATE_POPULAR)

	prod_path = os.path.join(CACHE_FOLDER, "productivities")
	if not os.path.exists(prod_path):
		os.makedirs(prod_path)
	year_wise_word_prod_mappings[year] = find_productivity(words=list(compass_unigrams.keys()), text=year_text, n=2, save_load_path=os.path.join(prod_path, f"{year}.json"))
	print()

UNIGRAM_KEYWORD_PLOT_SAVE_PATH = os.path.join(OUTPUT_FOLDER, "freq_norm_plots", "unigrams")
BIGRAM_KEYWORD_PLOT_SAVE_PATH = os.path.join(OUTPUT_FOLDER, "freq_norm_plots", "bigrams")
TRIGRAM_KEYWORD_PLOT_SAVE_PATH = os.path.join(OUTPUT_FOLDER, "freq_norm_plots", "trigrams")

if not os.path.exists(UNIGRAM_KEYWORD_PLOT_SAVE_PATH):
	os.makedirs(UNIGRAM_KEYWORD_PLOT_SAVE_PATH)
if not os.path.exists(BIGRAM_KEYWORD_PLOT_SAVE_PATH):
	os.makedirs(BIGRAM_KEYWORD_PLOT_SAVE_PATH)
if not os.path.exists(TRIGRAM_KEYWORD_PLOT_SAVE_PATH):
	os.makedirs(TRIGRAM_KEYWORD_PLOT_SAVE_PATH)

UNIGRAM_PROD_PLOT_SAVE_PATH = os.path.join(OUTPUT_FOLDER, "prod_plots", "unigrams")

if not os.path.exists(UNIGRAM_PROD_PLOT_SAVE_PATH):
	os.makedirs(UNIGRAM_PROD_PLOT_SAVE_PATH)

print(f"Making Frequency and Productivity Plots for Unigrams and Saving at {UNIGRAM_KEYWORD_PLOT_SAVE_PATH} AND {UNIGRAM_PROD_PLOT_SAVE_PATH}")
for word in compass_unigrams:
	plot_freq(word=word, year_wise_word_count_mappings=year_wise_unigram_mappings, save_path=os.path.join(UNIGRAM_KEYWORD_PLOT_SAVE_PATH,f"{word}.svg"), keyword_type=KEYWORD_METHOD)
	plot_prod(word=word, year_wise_prods_mappings=year_wise_word_prod_mappings, save_path=os.path.join(UNIGRAM_PROD_PLOT_SAVE_PATH, f"{word}.svg"))

print(f"Making Frequency Plots for Bigrams and Saving at {BIGRAM_KEYWORD_PLOT_SAVE_PATH}")
for word in compass_bigrams:
	plot_freq(word=word, year_wise_word_count_mappings=year_wise_bigram_mappings, save_path=os.path.join(BIGRAM_KEYWORD_PLOT_SAVE_PATH,f"{word}.svg"), keyword_type=KEYWORD_METHOD)

print(f"Making Frequency Plots for Trigrams and Saving at {TRIGRAM_KEYWORD_PLOT_SAVE_PATH}")
for word in compass_trigrams:
	plot_freq(word=word, year_wise_word_count_mappings=year_wise_trigram_mappings, save_path=os.path.join(TRIGRAM_KEYWORD_PLOT_SAVE_PATH,f"{word}.svg"), keyword_type=KEYWORD_METHOD)
print()
# keywords_for_sim_matrix_temp = list(compass_bigrams.keys())
# keywords_for_sim_matrix = []

# for ele in keywords_for_sim_matrix_temp:
# 	keywords_for_sim_matrix += ele.split(" ")
# keywords_for_sim_matrix = list(set(keywords_for_sim_matrix))
# # print(keywords_for_sim_matrix)

# sim_1 = compute_similarity_matrix_keywords(model_path="models/2017.model", keywords=keywords_for_sim_matrix)
# sim_2 = compute_similarity_matrix_keywords(model_path="models/2018.model", keywords=keywords_for_sim_matrix)

# acc = compute_acceleration_matrix(sim_1, sim_2)

# prs = top_k_acceleration(keywords_for_sim_matrix, acc, 40)
# top_unigrams = list(compass_unigrams.keys())
# plot_tsne("models/2017.model", (prs[0][0], prs[0][1]), top_unigrams, "1.svg")
# plot_tsne("models/2018.model", (prs[0][0], prs[0][1]), top_unigrams, "2.svg")