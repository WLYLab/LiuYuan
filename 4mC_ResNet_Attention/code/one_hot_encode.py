
import numpy as np
import itertools
import sys
def one_hot(seq):

    A = [1,0,0,0]
    T = [0,1,0,0]
    C = [0,0,1,0]
    G = [0,0,0,1]
    N = [0,0,0,0]
    feature_representation={"A":A,"C":C,"G":G,"T":T,"N":N}
    results = []

    for i in range(len(seq)):
        data_one_hot = []
        line_hot=[]
        for j in seq[i]:
            if j == "A":
                line_hot.extend(A)
            if j == "T":
                line_hot.extend(T)
            if j == "C":
                line_hot.extend(C)
            if j == "G":
                line_hot.extend(G)
            if j == "N":
                line_hot.extend(N)
        data_one_hot.append(line_hot)
        data_one_hot = np.array(data_one_hot)
        data_one_hot = data_one_hot.reshape((41,4))
        results.append(data_one_hot)
       
    results = np.array(results)
    return results