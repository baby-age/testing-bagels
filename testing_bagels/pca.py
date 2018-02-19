import pandas as pd
from sklearn.decomposition import PCA

def build_pca(data):
    pca = PCA(n_components=3)
    pca.fit(data)
    explained_variances = pca.explained_variance_ratio_

    components = pca.components_

    return explained_variances, components

def pca_transform(data):
	pca = PCA(n_components=2)
	transformed = pd.DataFrame(pca.fit_transform(data))

	return transformed