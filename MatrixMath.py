import numpy as np

w = [[1, 2], [3, 4]]
v = [[5, 6], [7, 8]]
i = [5, 6]

print(np.matmul(w, i), np.matmul(w, i).shape)
print(np.dot(w, i), np.dot(w, i).shape)
print(np.matmul(w, v))
print(np.dot(w, v))
