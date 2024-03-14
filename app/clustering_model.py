from sklearn.metrics import adjusted_rand_score
from sklearn.manifold import TSNE
from sklearn.cluster import AgglomerativeClustering, DBSCAN, KMeans, HDBSCAN
import pandas as pd
import matplotlib.pyplot as plt
from scipy import stats


class ClusterModel:
	def __init__(self,
				 n_clusters: int,
				 min_samples: int = 5,
				 min_cluster_size: int = 5,
				 metric: str = 'euclidean',
				 seed: int = 42,
				 eps: int = 0.5,
				 density: str = False
				 ):
		if density:
			self.model_1 = DBSCAN(eps=eps,
								  min_samples=min_samples,
								  metric=metric)
			self.model_2 = HDBSCAN(min_cluster_size=min_cluster_size,
								   metric=metric)
		else:
			self.model_1 = KMeans(n_clusters=n_clusters,
								  random_state=seed)
			self.model_2 = AgglomerativeClustering(n_clusters=n_clusters,
												   metric=metric,
												   linkage='single')

	def train(self, X) -> None:
		self.model_1.fit(X)
		self.model_2.fit(X)

	def fit(self, X) -> None:
		self.model_1.fit(X)
		self.model_2.fit(X)

	def validate_clusters(self, y_true) -> tuple[float, float]:
		model_1_rand = adjusted_rand_score(self.model_1.labels_, y_true)
		model_2_rand = adjusted_rand_score(self.model_2.labels_, y_true)

		return model_1_rand, model_2_rand


# def cluster_pieces(num_clusters, df):
# 	clustering_model = KMeans(n_clusters=num_clusters)
# 	clustering_model.fit(df.embedding)
# 	df['cluster_assignment'] = clustering_model.labels_
#
# 	return df