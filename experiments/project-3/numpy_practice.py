import numpy as np

# primitive 1 — matrix @ vector = dot product of EACH row with the vector
D = np.array([[1, 0], [0, 1], [1, 1]])   # 3 rows, each a 2-D "vector"
q = np.array([1, 1])
print(D @ q)                 # predict: ?

# primitive 2 — argsort returns the INDICES that would sort the list (ascending)
print(np.argsort([5, 2, 9]))  # predict: ?
