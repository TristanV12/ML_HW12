import numpy as np

def forward_propagation(X, S, W):
	for l in range(1, len(W)):
		S.append(np.dot(W[l].T, X[l - 1]))
		x_inner = [1]
		for i in np.nditer(S[-1]):
			if l != len(W) - 1:
				x_inner.append(np.tanh(i))
			else:
				X.append(np.tanh(i))
				return
		X.append(np.matrix(x_inner).T)

def back_propagation(X, Y, S, W, sigma):
	sigma = [1] * len(X)
	sigma[len(X) - 1] = 2 * (S[-1] - Y) * np.tanh(S[-1])
	for l in range(len(X) - 2, 0, -1):
		dtheta = 1 - np.multiply(X[l], X[l])[1:][:]
		sigma[l] = np.multiply(dtheta, np.dot(W[l + 1], (sigma[l + 1]))[1:][:])
	return sigma

def main():
	m = 2
	inputs = 2
	X = [np.matrix('1 '*(inputs + 1)).T]
	Y = 1
	S = [np.matrix('1 '*(inputs + 1)).T]
	W = [np.matrix('.2501 ' + '.25 '*(inputs)).T]
	sigma = []
	gradient = []
	for i in range(0, m):
		W.append(np.matrix([[.25]*inputs]*(inputs + 1)))
		sigma.append(np.matrix('.25 '*(inputs + 1)).T)
	W[1] = np.matrix('.2501 .25; .25 .25; .25 .25')
	W.append(np.matrix('.25 '*(inputs + 1)).T)
	forward_propagation(X, S, W)
	sigma = back_propagation(X, Y, S, W, sigma)
	print(X)
	print(W)
	for j in range(0, len(X) - 1):
		gradient.append(np.dot(X[j], sigma[j + 1].T))
	print(gradient)

if __name__ == '__main__':
	main()