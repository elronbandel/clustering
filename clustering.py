import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics.pairwise import cosine_similarity, euclidean_distances
from tqdm import tqdm
from sklearn.preprocessing import StandardScaler
from sklearn.utils import resample

from sklearn.decomposition import PCA
from sklearn.manifold import TSNE
from sklearn.manifold import Isomap
from sklearn.manifold import MDS
from umap import UMAP

from sklearn.neighbors import LocalOutlierFactor

from sklearn.cluster import KMeans, DBSCAN
from sknetwork.clustering import Louvain
from mst_clustering import MSTClustering
from sklearn.mixture import GaussianMixture

from sklearn.metrics import silhouette_samples, silhouette_score
from sklearn.metrics.cluster import v_measure_score

import matplotlib.pyplot as plt
import matplotlib.cm as cm
import numpy as np

class Identity:
    def __init__(self, *argv, **kwargs):
        pass
    def fit_transform(self, x, *argv, **kwargs):
        return x
class LouvainePredict(Louvain):
    def fit_predict(self, data,  *argv, **kwargs):
        dists = (cosine_similarity(data) + 1) / 2
        return self.fit_transform(dists, *argv, **kwargs)
class LocalOutlierFactorTransform(LocalOutlierFactor):
    def fit_transform(self, data,  *argv, **kwargs):
        labels = self.fit_predict(data)
        mask = labels != -1
        return data[mask, :]

def sort_args(*argv, **kwarg):
    return tuple(sorted(list(*argv) + list((k,v) for k, v in kwarg.items()))) 

def visualize2d(data, labels, dims=[0, 1]):
    d1, d2 = data[:, dims[0]], data[:, dims[1]]
    x_vis = pd.DataFrame({'d1': d1, 'd2':d2, 'label':labels}) 
    x_vis.plot.scatter('d1', 'd2', c='label', colormap='jet')

class ClusteringPipe:
    def __init__(self, data=None, labels=None, state=tuple() ,cache=dict(), sample=None):
        self.cache = cache
        self.state = state
        if data is not None:
            self.cache[state] = data
        self.labels = labels
        
        self._reduce = {
            'pca':PCA,
            'tsne':TSNE,
            'umap':UMAP,
            'non':Identity,
        }
        self._cluster = {
            'kmeans':KMeans,
            'dbscan':DBSCAN,
            'gmm':GaussianMixture,
            'louvain':LouvainePredict,
            'prim':MSTClustering,
            
        }
        self._score = {
            'silhouette':silhouette_samples,
            'v':v_measure_score,
        }
        self._avarage_score = {
            'silhouette':silhouette_score
        }
        self._normalize = {
            'std':StandardScaler,
        }
        self._remove_outlires = {
            'lof':LocalOutlierFactorTransform,
            'non':Identity,
        }
        
        for action in ['normalize', 'reduce', 'remove_outlires']:
            setattr(self, action, self.create_action(action, 'fit_transform'))
        for action in ['cluster']:
            setattr(self, action, self.create_action(action, 'fit_predict'))
#         for action in ['score', 'avarage_score']:
#             setattr(self, action, self.create_action(action))
        
        
    def data(self):
        return self.cache[self.state]
    
    def score(self, scoring, label=None, *argv, **kwargs):
        action, method = self.state[-1][:2]
        if action == 'cluster':
            data = self.cache[self.state_before()]
            if self.labels is not None and label:
                if label in self.labels:
                    data = self.labels[label]
            return self._score[scoring](data, self.data())
        
    def visualize(self, arg=None, *argv, **kwargs):
        action, method = self.state[-1][:2]
        if action == 'cluster':
            if arg in self._reduce:
                data = self.new(self.state_before()).reduce(arg, n_components=2).data()
                labels = self.data()
            elif self.labels is not None:
                if arg in self.labels.columns:
                    pass
            else:
                data = self.cache[self.state_before()]
                labels = self.data()
            visualize2d(data, labels)
        elif action == 'score':
            if method == 'silhouette':
                pass
    
    def create_action(self, action, function=None):
        def f(method, *argv, **kwarg):
            return self.apply(action, method, function, *argv, **kwarg)
        return f
    
    def state_before(self):
        return self.state[:-1]
    
    def new(self, state=tuple()):
        return ClusteringPipe(state=state, labels=self.labels, cache=self.cache)
        
    def apply(self, action, method, function, *argv, **kwarg):
        state = self.state + ((action, method) + sort_args(*argv, **kwarg),)
        if self.cache.get(state) is None:   
            action_dict = getattr(self, '_' + action)
            method_class = action_dict[method]
            method_instance = method_class(*argv, **kwarg)
            if function is not None:
                method_function = getattr(method_instance, function)
                predicted = method_function(self.data())
            else:
                predicted = method_instance
            self.cache[state] = predicted
        return self.new(state)
    
    
