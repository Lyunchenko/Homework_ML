import numpy as np

'''
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
'''

'''
q = np.array([1, 2, 2, 2, 1, 1, 3, 4, 4, 3])
print(q)
q = np.sort(q)[:-1]
print(q)

q = q[q>1]
print(q)
'''

'''
b = np.array([[2, 2, 3], [1, 5, 6], [7, 8, 9]])
print(b)
print('\n')
print(b[b[:,0]>1])
'''

'''
q = {'q':1, 'w': 2}
w = [{'e':1, 'r': 2}, {'q':1, 'w': 2}]

q = [q]
q.extend([w])
print(q)
'''

'''
b = np.array([[1, 2, 3], [0, 5, 6], [7, 8, 9]])
a = [10, 11, 12]
b = np.c_[b, a]

print(b)
b[:,0] = b[:,0]*2
print(b)
'''
'''
b = np.array([[2, 8, 3], 
	          [1, 5, 6], 
	          [7, 2, 9]])
print(b)

#q = np.sort(b, order=1, axis=0)
q = b[b[:,1].argsort()]

print(q)
'''

a = np.array([1, 2, 3, 4, 5])
b = np.array([[1, 1], [2, 1], [3, 1]])
print(b)

q = b[:,0]
print(q, a)

l1 = np.array(np.setdiff1d(a, q))
l2 = np.array([0 for x in range(len(l1))])
l = np.c_[l1, l2]
print(l1)
print(l2)
print(l)

c = np.append(b, l, axis=0)
print(c)
