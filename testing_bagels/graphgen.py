from numpy import random

def generate_graph_with_age(size, degree, max_weight, age):
	matrix = [[0 for x in range(size)] for x in range(size)]

	for i in range(len(matrix)):
		for j in range(i+1, len(matrix[i])):
			if random.uniform() < degree:
				weight = random.randint(1, max_weight)
				matrix[i][j] = weight
				matrix[j][i] = weight

	return matrix, age