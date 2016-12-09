import numpy as np
from numpy import linalg as LA

def forward_propagation(X, Y, S, W):
	for l in range(1, len(W)):
		S.append(np.dot(W[l].T, X[l - 1]))
		x_inner = [1]
		for i in np.nditer(S[-1]):
			if l != len(W) - 1:
				x_inner.append(np.tanh(i))
			else:
				X.append(np.tanh(i))
				if np.sign(X[-1]) != Y:
					return 1
				else:
					return 0
		X.append(np.matrix(x_inner).T)

def back_propagation(X, Y, S, W, sigma):
	sigma = [1] * len(X)
	sigma[len(X) - 1] = 2 * (S[-1] - Y) * np.tanh(S[-1])
	for l in range(len(X) - 2, 0, -1):
		dtheta = 1 - np.multiply(X[l], X[l])[1:][:]
		sigma[l] = np.multiply(dtheta, np.dot(W[l + 1], (sigma[l + 1]))[1:][:])
	return sigma

#function for finding symmetry
def sym(matrix):
	sym = 0
	maximum = 0
	for i in range(0,16):
		for j in range(0,8):
			sym = sym + abs(matrix[i][j] + matrix[i][15-j])
			maximum += 2
	return sym / maximum

#function for finding intensity
def intense(matrix):	
	total = 0
	maximum = 0
	for i in range(0,16):
		for j in range(0,16):
			total = total + matrix[i][j] + 1
			maximum += 2
	return total / maximum

def main():
	f = open("train.txt", "r")
	fwrite = open("output.txt", "w")
	m = 10
	inputs = 2
	iterations = 2 * (10 **6)
	X = []
	Y = []
	gradient = []

	for line in f:
		arr = line.split(' ')
		matrix = []
		inner = []
		new_x = [1] # [1 int sym]
		for a in range(1, len(arr)):
			if a % 16 == 1 and a != 1:
				matrix.append(inner)
				inner = []
			if(len(arr[a]) > 2):
				inner.append(float(arr[a]))
		matrix.append(inner)
		new_x.append(intense(matrix))
		new_x.append(sym(matrix))
		X.append([np.matrix(new_x).T])
		if(arr[0][0] == '1'):
			Y.append(1)
		else:
			Y.append(-1)

	S = [np.matrix('1 '*(inputs + 1)).T]
	W = [np.matrix('.025 ' + '.025 '*(inputs)).T]
	sigma = []
	for i in range(0, m):
		W.append(np.matrix([[.025]*inputs]*(inputs + 1)))
	sigma.append(np.matrix([[0.0]]))
	W[1] = np.matrix('.025 .025; .025 .025; .025 .025')
	W.append(np.matrix('.025 '*(inputs + 1)).T)

	for itr in range(0, iterations):
		for x in range(0, len(X)):
			for y in range(1, len(X[x])):
				del X[x][1]
		for g in range(0, len(gradient)):
			del gradient[0]
		gradient = []
		error = 0
		for i in range(0, m):
			gradient.append(np.matrix([[0.0]*inputs]*(inputs + 1)))
		gradient.append(np.matrix('0.0 '*(inputs + 1)).T)
		for x in range(0, len(X)):
			error += forward_propagation(X[x], Y[x], S, W)
			sigma = back_propagation(X[x], Y[x], S, W, sigma)
			for s in range(0, len(S)):
				del S[0]
			for j in range(0, len(X[x]) - 1):
				gradient[j] += np.dot(X[x][j], sigma[j + 1].T) / len(X)
		error /= len(X)
		for w in range(0, len(W) - 1):
			W[w + 1] -= .1 * gradient[w]
		if itr%100 == 0:
			print(itr, error)
		fwrite.write(str(itr) + " " + str(error) + "\n")
	print("W", W)
	print("error", error)

if __name__ == '__main__':
	main()