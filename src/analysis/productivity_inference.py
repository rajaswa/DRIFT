from src.analysis.tracking_clusters import kmeans_train
import numpy as np
import pandas as pd
def cluster_productivity(productivity_df, frequency_df):
	temp_df = productivity_df[['Word','Productivity']]
	temp_df['Frequency'] = frequency_df["Frequency"]
	input_dict = {}
	for row in temp_df.iterrows():
		if row[1]["Word"] in input_dict:
			input_dict[row[1]["Word"]] += [row[1]['Productivity'],row[1]["Frequency"]]
		else:
			input_dict[row[1]["Word"]] = [row[1]['Productivity'],row[1]["Frequency"]]
	final_array = []
	for keys in input_dict.keys():
		final_array.append(input_dict[keys])
	final_array = np.array(final_array)
	labels = kmeans_train(final_array, 3, method="sklearn")
	output_list = [[word, label] for word, label in zip(input_dict.keys(),labels)]
	output_df = pd.DataFrame(output_list,columns=["Words","Labels"])
	return output_df