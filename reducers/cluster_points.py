import numpy as np
from sklearn.cluster import DBSCAN
from .process_kwargs import process_kwargs
from flask import jsonify, request


DEFAULTS = {
    'eps': {'default': 5.0, 'type': float},
    'min_samples': {'default': 3, 'type': int},
    'metric': {'default': 'euclidean', 'type': str},
    'algorithm': {'default': 'auto', 'type': str},
    'leaf_size': {'default': 30, 'type': int},
    'p': {'default': None, 'type': float}
}


def process_data(data):
    # this will take the extracted data dumps and format them into an
    # array that DBSCAN can use
    # data comes in as [{tool1_x: 1, tool1_y: 2, tool2_x: 3, tool2_y: 4}, {tool1_x: 1, tool1_y: 2}, ...]
    unique_tools = set(sum([[k.split('_')[0] for k in d.keys()] for d in data], []))
    data_by_tool = {}
    for tool in unique_tools:
        tool_loc = []
        for d in data:
            x_key = '{0}_x'.format(tool)
            y_key = '{0}_y'.format(tool)
            if (x_key in d) and (y_key in d):
                tool_loc.append([d[x_key], d[y_key]])
        data_by_tool[tool] = tool_loc
    return data_by_tool


def cluster_points(data_by_tool, **kwargs):
    clusters = {}
    for tool, loc_list in data_by_tool.items():
        loc = np.array(loc_list)
        if loc.shape[0] > kwargs['min_samples']:
            db = DBSCAN(**kwargs).fit(np.array(loc))
            for k in set(db.labels_):
                if k > -1:
                    idx = db.labels_ == k
                    # number of points in the cluster
                    clusters['{0}_cluster{1}_count'.format(tool, k)] = idx.sum()
                    # mean of the cluster
                    k_loc = loc[idx].mean(axis=0)
                    clusters['{0}_cluster{1}_x'.format(tool, k)] = k_loc[0]
                    clusters['{0}_cluster{1}_y'.format(tool, k)] = k_loc[1]
                    # cov matrix of the cluster
                    k_cov = np.cov(loc[idx].T)
                    clusters['{0}_cluster{1}_var_x'.format(tool, k)] = k_cov[0, 0]
                    clusters['{0}_cluster{1}_var_y'.format(tool, k)] = k_cov[1, 1]
                    clusters['{0}_cluster{1}_var_x_y'.format(tool, k)] = k_cov[0, 1]
    return clusters


def process_request(request):
    data = process_data(request.get_json())
    kwargs = process_kwargs(request.args, DEFAULTS)
    return cluster_points(data, **kwargs)
