import bct
import testing_bagels.matrix_preprocessor as mp

'''
Calculate modularity and efficiency coefficients with Brain Connectivity Toolbox.
Input should be data frame, output is list.
Zip if used together for prediction, like this: X = [list(x) for x in zip(mod, eff)]
'''
def modularity_and_efficiency(data):
	mod_scores = []
	eff_scores = []
	for i, x in enumerate(data):
	    matrix = mp.preprocess_matrix(x)
	    mod_score = bct.modularity_und(matrix)[1]
	    eff_score = bct.efficiency_wei(matrix)
	    
	    mod_scores.append(mod_score)
	    eff_scores.append(eff_score)

	return mod_scores, eff_scores

'''
Calculate local clustering coefficients with Brain Connectivity Toolbox.
Input should be data frame, output is list.
'''
def local_clustering(data):
	cluster_coeffs = []
	for i, x in enumerate(data):
	    matrix = mp.preprocess_matrix(x)
	    coeffs = bct.clustering_coef_wu(matrix)
	    
	    cluster_coeffs.append(coeffs.tolist())

	return cluster_coeffs
