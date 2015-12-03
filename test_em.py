import datetime
import numpy as np
import random
import sys
import time

import matplotlib.pyplot as plt

import matplotlib.image as img
import matplotlib.cm as cm

from em import run_em as fast_run_em


def generate_test_data_fixed_face(H, W, N, s):
    face = [
        [0,0,0,0,0,0,0],
        [0,100,100,100,100,100,0],
        [0,100,33,100,33,100,0],
        [0,100,100,100,100,100,0],
        [0,100,200,200,200,100,0],
        [0,0,100,100,100,0,0],
        [0,0,0,0,0,0,0]]
    h, w = 7, 7
    # bcg = np.fromfunction(lambda i, j: (i * 67) % 255, (H, W))
    bcg = np.fromfunction(lambda i, j: 33, (H, W))
    X = np.zeros((H, W, N), dtype=np.float)
    for k in range(N):
        st_x = int(np.random.uniform(H-h+1))
        st_y = int(np.random.uniform(W-w+1))
        X[:, :, k] = bcg
        X[st_x:st_x + h, st_y:st_y + w, k] = face
    noise = np.random.normal(0, s, (H, W, N))
    return X + noise


def main():
	H, W, N = 15, 10, 50
	s = 1
	X = generate_test_data_fixed_face(H, W, N, s)
	h, w = 7, 7

	F = np.random.uniform(0, 255, (h, w))
	# F = np.fromfunction(lambda i, j: ((i + j) * 100) % 255, (h, w))

	B = np.random.uniform(0, 255, (H, W))
	# B = np.fromfunction(lambda i, j: ((i + j) * 10) % 255, (H, W))
	# A = np.random.uniform(0, 255, ((H-h+1), (W-w+1)))
	A = np.ones(((H-h+1), (W-w+1)))
	# A = np.fromfunction(lambda i, j: ((i + j) * 50) % 255, ((H-h+1), (W-w+1)))
	A /= np.sum(A)
	s = 1
	plt.imshow(B, cmap = cm.Greys_r, interpolation='None')
	plt.savefig('B1.png')
	plt.imshow(F, cmap = cm.Greys_r, interpolation='None')
	plt.savefig('F1.png')
	plt.imshow(A, cmap = cm.Greys_r, interpolation='None')
	plt.savefig('A1.png')

	print("generated!")
	r = fast_run_em(X, h, w, F, B, s, A, 20)
	print("returned!")

	# r = fast_run_em(X, h, w, F, B, s, A, 10)
	# r = fast_run_em(h, w, s, 10)
	# r = fast_run_em(X, h, w, s, 10)
	# print(r)
	F2, B2, s2, A2 = r
	plt.imshow(np.nan_to_num(B2), cmap = cm.Greys_r, interpolation='None')
	plt.savefig('B2.png')
	plt.imshow(np.nan_to_num(F2), cmap = cm.Greys_r, interpolation='None')
	plt.savefig('F2.png')
	plt.imshow(np.nan_to_num(A2), cmap = cm.Greys_r, interpolation='None')
	plt.savefig('A2.png')

    # res, time = check_time(is_connected, test_graph, 0)
    # print "Python:", res, time
    # res, time = check_time(fast_is_connected, test_graph, 0)
    # print "C:", res, time


if __name__ == '__main__':
    main()

