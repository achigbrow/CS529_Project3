from sklearn.decomposition import PCA
from sklearn.preprocessing import MinMaxScaler


def get_pca(features):
  pca = PCA(n_components=2)
  transformed = pca.fit(features).transform(features)
  scaler = MinMaxScaler()
  scaler.fit(transformed)
  return scaler.transform(transformed)


my_array_of_feature_vectors = ...
scaled_pca = get_pca(my_array_of_feature_vectors)