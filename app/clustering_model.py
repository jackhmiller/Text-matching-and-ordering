from sklearn.metrics import adjusted_rand_score
from sklearn.cluster import AgglomerativeClustering, DBSCAN, KMeans, HDBSCAN
from sklearn.preprocessing import normalize


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
		self.density = density
		if self.density:
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
		if not self.density:
			X = normalize(X)
			# Since KMeans measures Euclidean distance (which is proportional to cosine similarity)
			# normalizing the vector embeddings means we are using dot product which is equal to cosine similarity
		self.model_1.fit(X)
		self.model_2.fit(X)

	def validate_clusters(self, y_true) -> tuple[float, float]:
		model_1_rand = adjusted_rand_score(self.model_1.labels_, y_true)
		model_2_rand = adjusted_rand_score(self.model_2.labels_, y_true)

		return model_1_rand, model_2_rand
