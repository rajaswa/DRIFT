import matplotlib.pyplot as plt
import numpy as np
from scipy.spatial import ConvexHull, Delaunay, KDTree


def get_contour_matrix(kmeans, embeds, method, steps=500):
    x_one_perc = (embeds[:, 0].max() - embeds[:, 0].min()) * 0.1
    y_one_perc = (embeds[:, 1].max() - embeds[:, 1].min()) * 0.1
    x_range = np.linspace(
        embeds[:, 0].min() - x_one_perc, embeds[:, 0].max() + x_one_perc, num=steps
    )
    y_range = np.linspace(
        embeds[:, 1].min() - y_one_perc, embeds[:, 1].max() + y_one_perc, num=steps
    )

    X, Y = np.meshgrid(x_range, y_range)
    X = X.ravel().reshape(-1, 1)
    Y = Y.ravel().reshape(-1, 1)
    points = np.concatenate([X, Y], axis=1)

    if method == "faiss":
        Z = kmeans.index.search(points.astype(np.float32), 1)[1]
    elif method == "sklearn":
        Z = kmeans.predict(points)

    Z = np.array(Z)
    Z = Z.reshape(steps, steps)
    return x_range, y_range, Z


# TO-DO: To be fixed
def triangle_area(p1, p2, p3):
    return abs(
        (p1[0] * (p2[1] - p3[1]) + p2[0] * (p3[1] - p1[1]) + p3[0] * (p1[1] - p2[1]))
        / 2.0
    )


def within_triangle(p1, p2, p3, p):
    area = triangle_area(p1, p2, p3)
    area1 = triangle_area(p, p2, p3)
    area2 = triangle_area(p1, p, p3)
    area3 = triangle_area(p1, p2, p)

    if abs(area - area1 - area2 - area3) < 1e-3:
        return True
    else:
        return False


def within_polygon(polygon_points, polygon_area, p):
    area = 0
    for idx in range(0, len(polygon_points)):
        if idx < len(polygon_points) - 1:
            p1, p2 = polygon_points[idx], polygon_points[idx + 1]
            area += triangle_area(p1, p2, p)
        else:
            p1, p2 = polygon_points[idx], polygon_points[0]
            area += triangle_area(p1, p2, p)
    if abs(polygon_area - area) < 1e-3:
        return True
    else:
        return False


def get_contour_matrix_using_kdtree(embeds, labels, dist_bound=25, steps=50):
    x_one_perc = (embeds[:, 0].max() - embeds[:, 0].min()) * 0.1
    y_one_perc = (embeds[:, 1].max() - embeds[:, 1].min()) * 0.1
    x_range = np.linspace(
        embeds[:, 0].min() - x_one_perc, embeds[:, 0].max() + x_one_perc, num=steps
    )
    y_range = np.linspace(
        embeds[:, 1].min() - y_one_perc, embeds[:, 1].max() + y_one_perc, num=steps
    )

    X, Y = np.meshgrid(x_range, y_range)
    X = X.ravel().reshape(-1, 1)
    Y = Y.ravel().reshape(-1, 1)
    points = np.concatenate([X, Y], axis=1)

    Z = []
    tree = KDTree(embeds)
    distances, indices = tree.query(points, k=3)

    for loop_idx, (dist, idx) in enumerate(zip(distances, indices)):
        p1, p2, p3 = embeds[idx[0]], embeds[idx[1]], embeds[idx[2]]
        print(p1, p2, p3)
        p = points[loop_idx]
        if within_triangle(p1, p2, p3, p):
            unique_labels, counts = np.unique(
                [labels[idx[0]], labels[idx[1]], labels[idx[2]]], return_counts=True
            )
            for label, count in zip(unique_labels, counts):
                if count >= 2:
                    Z.append(label)
                    break
            else:
                Z.append(labels[idx][0])
        else:
            if dist[0] <= dist_bound:
                Z.append(labels[idx[0]])
            else:
                Z.append(-1)

    Z = np.array(Z)
    Z = Z.reshape(steps, steps)
    return x_range, y_range, Z


def get_contour_matrix_using_triangulation(embeds, labels, steps=500):
    x_one_perc = (embeds[:, 0].max() - embeds[:, 0].min()) * 0.1
    y_one_perc = (embeds[:, 1].max() - embeds[:, 1].min()) * 0.1
    x_range = np.linspace(
        embeds[:, 0].min() - x_one_perc, embeds[:, 0].max() + x_one_perc, num=steps
    )
    y_range = np.linspace(
        embeds[:, 1].min() - y_one_perc, embeds[:, 1].max() + y_one_perc, num=steps
    )

    X, Y = np.meshgrid(x_range, y_range)
    X = X.ravel().reshape(-1, 1)
    Y = Y.ravel().reshape(-1, 1)
    points = np.concatenate([X, Y], axis=1)

    Z = {}
    tri = Delaunay(embeds)
    for indice_triple in tri.simplices:
        p1, p2, p3 = (
            embeds[indice_triple[0]],
            embeds[indice_triple[1]],
            embeds[indice_triple[2]],
        )
        l1, l2, l3 = (
            labels[indice_triple[0]],
            labels[indice_triple[1]],
            labels[indice_triple[2]],
        )
        if l1 == l2 and l2 == l3:
            for point_idx, point in enumerate(points):
                if point_idx in Z and Z[point_idx] != -1:
                    continue
                else:
                    if within_triangle(p1, p2, p3, point):
                        Z[point_idx] = l1
                    else:
                        Z[point_idx] = -1

    Z_values = [v for k, v in sorted(Z.items(), key=lambda x: x[0])]
    Z = np.array(list(Z_values))
    Z = Z.reshape(steps, steps)
    return x_range, y_range, Z


def get_contour_matrix_kdree_and_triangulation(embeds, labels, dist_bound=2, steps=500):
    x_one_perc = (embeds[:, 0].max() - embeds[:, 0].min()) * 0.1
    y_one_perc = (embeds[:, 1].max() - embeds[:, 1].min()) * 0.1
    x_range = np.linspace(
        embeds[:, 0].min() - x_one_perc, embeds[:, 0].max() + x_one_perc, num=steps
    )
    y_range = np.linspace(
        embeds[:, 1].min() - y_one_perc, embeds[:, 1].max() + y_one_perc, num=steps
    )

    X, Y = np.meshgrid(x_range, y_range)
    X = X.ravel().reshape(-1, 1)
    Y = Y.ravel().reshape(-1, 1)
    points = np.concatenate([X, Y], axis=1)

    Z = {}
    tri = Delaunay(embeds)
    for indice_triple in tri.simplices:
        p1, p2, p3 = (
            embeds[indice_triple[0]],
            embeds[indice_triple[1]],
            embeds[indice_triple[2]],
        )
        l1, l2, l3 = (
            labels[indice_triple[0]],
            labels[indice_triple[1]],
            labels[indice_triple[2]],
        )
        if l1 == l2 and l2 == l3:
            for point_idx, point in enumerate(points):
                if point_idx in Z and Z[point_idx] != -1:
                    continue
                else:
                    if within_triangle(p1, p2, p3, point):
                        Z[point_idx] = l1
                    else:
                        Z[point_idx] = -1

    Z_values = [v for k, v in sorted(Z.items(), key=lambda x: x[0])]
    Z = np.array(list(Z_values))

    tree = KDTree(embeds)
    distances, indices = tree.query(points, k=1)

    for loop_idx, (dist, idx) in enumerate(zip(distances, indices)):
        if dist <= dist_bound and Z[loop_idx] == -1:
            Z[loop_idx] = labels[idx]
    Z = Z.reshape(steps, steps)
    return x_range, y_range, Z


def get_contour_matrix_convex_hull(embeds, labels, steps=500):
    x_one_perc = (embeds[:, 0].max() - embeds[:, 0].min()) * 0.1
    y_one_perc = (embeds[:, 1].max() - embeds[:, 1].min()) * 0.1
    x_range = np.linspace(
        embeds[:, 0].min() - x_one_perc, embeds[:, 0].max() + x_one_perc, num=steps
    )
    y_range = np.linspace(
        embeds[:, 1].min() - y_one_perc, embeds[:, 1].max() + y_one_perc, num=steps
    )

    X, Y = np.meshgrid(x_range, y_range)
    X = X.ravel().reshape(-1, 1)
    Y = Y.ravel().reshape(-1, 1)
    points = np.concatenate([X, Y], axis=1)

    label_to_point_map = {}
    for idx, label in enumerate(labels):
        if label not in label_to_point_map:
            label_to_point_map[label] = [embeds[idx]]
        else:
            label_to_point_map[label] += [embeds[idx]]

    Z = {}
    for label, label_points in label_to_point_map.items():
        label_points = np.array(label_points)
        hull = ConvexHull(label_points)
        for point_idx, point in enumerate(points):
            if point_idx in Z and Z[point_idx] != -1:
                continue
            else:
                if within_polygon(label_points[hull.vertices], hull.volume, point):
                    Z[point_idx] = label
                else:
                    Z[point_idx] = -1

    Z_values = [v for k, v in sorted(Z.items(), key=lambda x: x[0])]
    Z = np.array(list(Z_values))

    Z = Z.reshape(steps, steps)
    return x_range, y_range, Z
