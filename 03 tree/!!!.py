import numpy as np

a = [10, 11, 12]
b = np.array([[1, 2, 3], [0, 5, 6], [7, 8, 9]])

print(b)

b = np.append(b, [a], axis=0)

a = [10, 11, 12, 13]
b = np.c_[b, a]
b = np.c_[b, a]

print(b)

print('\nShape')
q = np.array([1, 2, 2, 2, 1, 1, 3])
print(q.shape[0])

print(b[2:])

print('\nSotr')
print(b[b[:,0].argsort()])


print('\nUnic value')
q = np.array([1, 2, 2, 2, 1, 1, 3])
print(q)
unique_rows = np.unique(q, axis=0)

print(unique_rows)


print('\nValue count')
q = np.array([0, 2, 2, 2, 1, 1, 3])
unique_val, val_count = np.unique(q, return_counts=True, axis=0)
print(unique_val, val_count)
arr = np.c_[unique_val, val_count]
print(arr)

dtype = dict.fromkeys([1, 2], (10, 20))
print(dtype)
