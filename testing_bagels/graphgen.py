from numpy import random
import pandas as pand
import testing_bagels.embedder as emb
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D


def generate_graph_with_age(size, degree, max_weight, age):
	matrix = [[0 for x in range(size)] for x in range(size)]

	for i in range(len(matrix)):
		for j in range(i+1, len(matrix[i])):
			if random.uniform() < degree:
				weight = random.randint(1, max_weight)
				matrix[i][j] = weight
				matrix[j][i] = weight

	return matrix, age

def generate_graphs(number, dimension, max_weight):
	data = []
	for i in range(number):
		ran = random.uniform(0, 1)
		if ran < 0.33:
			age = 12
		elif ran < 0.66:
			age = 14
		else:
			age = 16

		data.append(generate_graph_with_age(dimension, ran, max_weight, age))
	df = pand.DataFrame(data, columns = ['X', 'y'])
	return(df)
