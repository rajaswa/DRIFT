import inspect
import os
import re
from src.analysis.similarity_acc_matrix import compute_acc_between_years
import numpy as np
import pandas as pd
import math
import matplotlib.pyplot as plt

from src.utils.statistics import find_freq, find_norm_freq, find_productivity
from scipy.spatial import KDTree, Delaunay, ConvexHull
def get_default_args(func):
    signature = inspect.signature(func)
    return {
        k: v.default
        for k, v in signature.parameters.items()
        if v.default is not inspect.Parameter.empty
    }


def get_values_from_indices(lst, idx_list):
    return [lst[idx] for idx in idx_list], [
        lst[idx] for idx in range(len(lst)) if idx not in idx_list
    ]


def get_years_from_data_path(data_path):
    years = sorted(
        [fil.split(".")[0] for fil in os.listdir(data_path) if fil != "compass.txt"]
    )
    return years


def get_productivity_for_range(
    start_year, end_year, selected_ngrams, years, data_path, n
):
    yearss = []
    words = []
    prodss = []
    start_year_idx = years.index(start_year)
    end_year_idx = years.index(end_year)
    for year_idx in range(start_year_idx, end_year_idx + 1):
        year = years[year_idx]
        year_text = read_text_file(data_path, year)
        prods = find_productivity(selected_ngrams, year_text, n)
        for word, productivity in prods.items():
            yearss.append(year)
            words.append(word)
            prodss.append(productivity)
    productivity_df = pd.DataFrame.from_dict(
        {"Year": yearss, "Word": words, "Productivity": prodss}
    )
    return productivity_df

def get_frequency_for_range(
    start_year, end_year, selected_ngrams, years, data_path, n, normalize=False
):
    yearss = []
    words = []
    freqss = []
    start_year_idx = years.index(start_year)
    end_year_idx = years.index(end_year)
    for year_idx in range(start_year_idx, end_year_idx + 1):
        year = years[year_idx]
        year_text = read_text_file(data_path, year)
        if normalize:
            freqs = find_norm_freq(year_text, n=n, sort=False)
        else:
            freqs = find_freq(year_text, n=n, sort=False)
        for word in selected_ngrams:
            yearss.append(year)
            words.append(word)
            freqss.append(freqs[word] if word in freqs else 0)
    frequency_df = pd.DataFrame.from_dict(
        {"Year": yearss, "Word": words, "Frequency": freqss}
    )
    return frequency_df


def get_acceleration_bw_models(
    year1, year2, model_path, selected_ngrams, all_model_vectors, top_k_acc
):
    model_path1 = os.path.join(model_path, year1 + ".model")
    model_path2 = os.path.join(model_path, year2 + ".model")

    word_pairs, em1, em2 = compute_acc_between_years(
        selected_ngrams,
        model_path1,
        model_path2,
        all_model_vectors=all_model_vectors,
        top_k_acc=top_k_acc,
        skip_same_word_pairs=True,
        skip_duplicates=True,
    )
    return word_pairs, em1, em2


def get_word_pair_sim_bw_models(
    year1, year2, model_path, selected_ngrams, all_model_vectors, top_k_acc
):
    word_pairs, em1, em2 = get_acceleration_bw_models(
        year1, year2, model_path, selected_ngrams, all_model_vectors, top_k_acc
    )
    word_pair_sim_df = pd.DataFrame(
        list(word_pairs.items()), columns=["Word Pair", "Acceleration"]
    )
    word_pair_sim_df = word_pair_sim_df.sort_values(by="Acceleration", ascending=False)

    word_pair_sim_df_words = []
    for word1, word2 in word_pair_sim_df["Word Pair"].values:
        if word1 not in word_pair_sim_df_words:
            word_pair_sim_df_words.append(word1)
        if word2 not in word_pair_sim_df_words:
            word_pair_sim_df_words.append(word2)
    return word_pair_sim_df, word_pair_sim_df_words


def read_text_file(data_path, name):
    with open(os.path.join(data_path, name + ".txt"), encoding="utf-8") as f:
        words = f.read()
    return words

def get_contour_matrix(kmeans, embeds, method, steps=500):
    x_one_perc = (embeds[:, 0].max()-embeds[:, 0].min())*0.1
    y_one_perc = (embeds[:, 1].max()-embeds[:, 1].min())*0.1    
    x_range = np.linspace(embeds[:, 0].min()-x_one_perc,  embeds[:, 0].max()+x_one_perc, num=steps)
    y_range = np.linspace(embeds[:, 1].min()-y_one_perc,  embeds[:, 1].max()+y_one_perc, num=steps)

    X, Y = np.meshgrid(x_range, y_range)
    X = X.ravel().reshape(-1,1)
    Y = Y.ravel().reshape(-1,1)
    points = np.concatenate([X,Y],axis=1)
    if method == "faiss":
        Z = kmeans.index.search(points.astype(np.float32), 1)[1]
    elif method == "sklearn":
        Z = kmeans.predict(points)

    Z = np.array(Z)
    Z = Z.reshape(steps, steps)
    return x_range,y_range,Z
    
# TO-DO: To be fixed
def triangle_area(p1, p2, p3):
    return abs((p1[0]*(p2[1]-p3[1])+p2[0]*(p3[1]-p1[1])+p3[0]*(p1[1]-p2[1]))/2.0)

def within_triangle(p1, p2, p3, p):
    area = triangle_area(p1,p2,p3)
    area1 = triangle_area(p, p2, p3)
    area2 = triangle_area(p1, p, p3)
    area3 = triangle_area(p1, p2, p)

    if abs(area-area1-area2-area3)<1e-3:
        return True
    else:
        return False

def within_polygon(polygon_points, polygon_area, p):
    area = 0
    for idx in range(0, len(polygon_points)):
        if idx<len(polygon_points)-1:
            p1, p2 = polygon_points[idx], polygon_points[idx+1]
            area+=triangle_area(p1, p2, p)
        else:
            p1, p2 = polygon_points[idx], polygon_points[0]
            area+=triangle_area(p1, p2, p)
    if abs(polygon_area-area)<1e-3:
        return True
    else:
        return False

def get_contour_matrix_using_kdtree(embeds,labels, dist_bound = 25, steps=50):
    x_one_perc = (embeds[:, 0].max()-embeds[:, 0].min())*0.1
    y_one_perc = (embeds[:, 1].max()-embeds[:, 1].min())*0.1    
    x_range = np.linspace(embeds[:, 0].min()-x_one_perc,  embeds[:, 0].max()+x_one_perc, num=steps)
    y_range = np.linspace(embeds[:, 1].min()-y_one_perc,  embeds[:, 1].max()+y_one_perc, num=steps)

    X, Y = np.meshgrid(x_range, y_range)
    X = X.ravel().reshape(-1,1)
    Y = Y.ravel().reshape(-1,1)
    points = np.concatenate([X,Y],axis=1)

    Z = []
    tree = KDTree(embeds)
    distances, indices = tree.query(points, k=3)
    
    for loop_idx, (dist, idx) in enumerate(zip(distances, indices)):
        p1, p2, p3 = embeds[idx[0]], embeds[idx[1]], embeds[idx[2]]
        print(p1, p2, p3)
        p = points[loop_idx]
        if within_triangle(p1,p2,p3, p):
            unique_labels, counts = np.unique([labels[idx[0]],labels[idx[1]],labels[idx[2]]], return_counts=True)
            for label,count in zip(unique_labels,counts):
                if count>=2:
                    Z.append(label)
                    break
            else:
                Z.append(labels[idx][0])
        else:
            if dist[0]<=dist_bound:
                Z.append(labels[idx[0]])
            else:
                Z.append(-1)

    Z = np.array(Z)
    Z = Z.reshape(steps, steps)
    return x_range,y_range,Z


def get_contour_matrix_using_triangulation(embeds,labels, steps=500):
    x_one_perc = (embeds[:, 0].max()-embeds[:, 0].min())*0.1
    y_one_perc = (embeds[:, 1].max()-embeds[:, 1].min())*0.1    
    x_range = np.linspace(embeds[:, 0].min()-x_one_perc,  embeds[:, 0].max()+x_one_perc, num=steps)
    y_range = np.linspace(embeds[:, 1].min()-y_one_perc,  embeds[:, 1].max()+y_one_perc, num=steps)

    X, Y = np.meshgrid(x_range, y_range)
    X = X.ravel().reshape(-1,1)
    Y = Y.ravel().reshape(-1,1)
    points = np.concatenate([X,Y],axis=1)

    Z = {}
    tri = Delaunay(embeds)
    for indice_triple in tri.simplices:
        p1, p2, p3 = embeds[indice_triple[0]], embeds[indice_triple[1]], embeds[indice_triple[2]]
        l1, l2, l3 = labels[indice_triple[0]], labels[indice_triple[1]], labels[indice_triple[2]]
        if l1==l2 and l2==l3:
            for point_idx, point in enumerate(points):
                if point_idx in Z and Z[point_idx]!=-1:
                    continue
                else:
                    if within_triangle(p1,p2,p3, point):
                        Z[point_idx]=l1
                    else:
                        Z[point_idx]=-1
    
    Z_values = [v for k, v in sorted(Z.items(), key=lambda x: x[0])]
    Z = np.array(list(Z_values))
    Z = Z.reshape(steps, steps)
    return x_range,y_range,Z

def get_contour_matrix_kdree_and_triangulation(embeds,labels,dist_bound=2, steps=500):
    x_one_perc = (embeds[:, 0].max()-embeds[:, 0].min())*0.1
    y_one_perc = (embeds[:, 1].max()-embeds[:, 1].min())*0.1    
    x_range = np.linspace(embeds[:, 0].min()-x_one_perc,  embeds[:, 0].max()+x_one_perc, num=steps)
    y_range = np.linspace(embeds[:, 1].min()-y_one_perc,  embeds[:, 1].max()+y_one_perc, num=steps)

    X, Y = np.meshgrid(x_range, y_range)
    X = X.ravel().reshape(-1,1)
    Y = Y.ravel().reshape(-1,1)
    points = np.concatenate([X,Y],axis=1)

    Z = {}
    tri = Delaunay(embeds)
    for indice_triple in tri.simplices:
        p1, p2, p3 = embeds[indice_triple[0]], embeds[indice_triple[1]], embeds[indice_triple[2]]
        l1, l2, l3 = labels[indice_triple[0]], labels[indice_triple[1]], labels[indice_triple[2]]
        if l1==l2 and l2==l3:
            for point_idx, point in enumerate(points):
                if point_idx in Z and Z[point_idx]!=-1:
                    continue
                else:
                    if within_triangle(p1,p2,p3, point):
                        Z[point_idx]=l1
                    else:
                        Z[point_idx]=-1
    
    Z_values = [v for k, v in sorted(Z.items(), key=lambda x: x[0])]
    Z = np.array(list(Z_values))


    tree = KDTree(embeds)
    distances, indices = tree.query(points, k=1)
    
    for loop_idx, (dist, idx) in enumerate(zip(distances, indices)):
        if dist<=dist_bound and Z[loop_idx]==-1:
            Z[loop_idx]=labels[idx]
    Z = Z.reshape(steps, steps)
    return x_range,y_range,Z

def get_contour_matrix_convex_hull(embeds, labels, steps=500):
    x_one_perc = (embeds[:, 0].max()-embeds[:, 0].min())*0.1
    y_one_perc = (embeds[:, 1].max()-embeds[:, 1].min())*0.1    
    x_range = np.linspace(embeds[:, 0].min()-x_one_perc,  embeds[:, 0].max()+x_one_perc, num=steps)
    y_range = np.linspace(embeds[:, 1].min()-y_one_perc,  embeds[:, 1].max()+y_one_perc, num=steps)

    X, Y = np.meshgrid(x_range, y_range)
    X = X.ravel().reshape(-1,1)
    Y = Y.ravel().reshape(-1,1)
    points = np.concatenate([X,Y],axis=1)

    label_to_point_map = {}
    for idx, label in enumerate(labels):
        if label not in label_to_point_map:
            label_to_point_map[label]=[embeds[idx]]
        else:
            label_to_point_map[label]+=[embeds[idx]]

    Z = {}
    for label, label_points in label_to_point_map.items():
        label_points = np.array(label_points)
        hull = ConvexHull(label_points)
        for point_idx, point in enumerate(points):
            if point_idx in Z and Z[point_idx]!=-1:
                continue
            else:
                if within_polygon(label_points[hull.vertices], hull.volume, point):
                    Z[point_idx]=label
                else:
                    Z[point_idx]=-1

    
   
    Z_values = [v for k, v in sorted(Z.items(), key=lambda x: x[0])]
    Z = np.array(list(Z_values))

    Z = Z.reshape(steps, steps)
    return x_range,y_range,Z