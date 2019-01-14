
# coding: utf-8

import pandas as pd
import json
import os 


data = []
with open('data/pubs_test.json') as f:
    data = json.load(f)


def get_co_author(data_it):
    authors = []
    for author in data_it['authors']:
        authors.append(author['name'].replace(' ', '').replace('-', ''))
    return ' '.join(authors)

def generate_output_one_set(labels, data, size):
    print('generate output')
    output_set = []
    i = 0
    while i < size:
        output_set.append([])
        i += 1
    i = 0
    for it in labels:
        if (it != -1):
            output_set[it].append(data[i]['id'])
        i += 1
    return output_set

from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.cluster import DBSCAN
from sklearn import metrics
import numpy as np

global_result = {}
for set_d in data:
    print(set_d)
    vectorizer = TfidfVectorizer()
    docs = []
    for it in data[set_d]:
        docs.append(get_co_author(it))
    X = vectorizer.fit_transform(docs)
    #print(vectorizer.get_feature_names())
    db = DBSCAN(eps=0.6, min_samples=3, metric="cosine").fit(X)
    ##
    core_samples_mask = np.zeros_like(db.labels_, dtype=bool)
    core_samples_mask[db.core_sample_indices_] = True
    labels = db.labels_

    n_clusters_ = len(set(labels)) - (1 if -1 in labels else 0)
    n_noise_ = list(labels).count(-1)
    print('cluster OK:', n_clusters_)
    print('cluster noise:', n_noise_)
    
    output_set = generate_output_one_set(labels, data[set_d], n_clusters_)
    global_result[set_d] = output_set


json.dumps(global_result)


with open('result.json', 'w') as of:
    json.dump(global_result, of)

