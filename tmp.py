import numpy as np
n_context = 33
dim = 22
# Q = np.random.normal(size=(n_context, dim))
# K = np.random.normal(size=(n_context, dim))
# S = np.zeros(shape=(n_context, n_context))



# for i in range(n_context):
#     for j in range(n_context):
#         for k in range(dim):
#             S[i][j] += Q[i][k] * K[j][k]
# print(S[:5, :5])

# print(np.dot(Q, K.T)[:5, :5])

P = np.random.normal(size=(n_context, n_context))
V = np.random.normal(size=(n_context, dim))
O = np.zeros(shape=(n_context, dim))

for i in range(thread_id):
    for j in range(n_context):
        for k in range(dim):
            O[i][k] += P[i][j] * V[j][k]

print(np.sum(O - np.matmul(P, V)))