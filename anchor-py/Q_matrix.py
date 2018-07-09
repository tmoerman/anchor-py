from __future__ import division
import numpy as np
import time
import math
from helper_functions import *
from scipy.sparse.csc import csc_matrix


def generate_Q_matrix(M: csc_matrix) -> np.ndarray:
    """
    Candidate for a distributed implementation?

    :param M: a sparse CSC document matrix M (with floating point entries)
    :return: the word-word correlation matrix Q as a dense Numpy ndarray.
    """

    vocabulary_size, n_docs = M.shape

    diag_M = np.zeros(vocabulary_size)
    for j in range(M.indptr.size - 1):
        # start and end indices for column j
        start = M.indptr[j]
        end = M.indptr[j + 1]
        
        words_per_doc = np.sum(M.data[start:end])

        row_indices = M.indices[start:end]
        
        diag_M[row_indices] += M.data[start:end] / (words_per_doc * (words_per_doc-1))
        M.data[start:end] = M.data[start:end] / math.sqrt(words_per_doc * (words_per_doc-1))

    Q = M*M.transpose() / n_docs
    Q = Q.todense()
    Q = np.array(Q, copy=False)

    diag_M = diag_M / n_docs
    Q = Q - np.diag(diag_M)
    
    # print('Sum of entries in Q is ', np.sum(Q))

    return Q
