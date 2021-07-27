```python
import numpy as np
import matplotlib.pyplot as plt

def load_turing(path):
    with open(path, "r") as f:
        im = np.zeros((600, 800), dtype = np.int64)
        im_raw = f.readlines()
        for i in range(1, 601):
            for j in range(800):
                if im_raw[i][j] == "#":
                    im[i-1, j] = 1
    return im

im = load_turing("turing.txt")
plt.imshow(im);
im.sum()  # number of ones to paint
```
```python
from numba import jit # todo

def max_square(M):  # return a square of 1 with maximum size, by DP
    P = M.copy()  # P[i, j] will be the max size of a square with bottom left (i, j)
    for i in range(1, len(M)):
        for j in range(1, len(M[0])):
            if M[i, j] == 1:
                P[i, j] = min([P[i-1, j], P[i-1, j-1], P[i, j-1]]) + 1
    pos = np.unravel_index(np.argmax(P, axis=None), P.shape)
    return pos, P[pos]

def fill(i, j, k):
    return "FILL," + str(j) + "," + str(i) + "," + str(k) + "\n" 

def greedy(M):
    with open("operations.txt", "w") as f:
        k = 2
        while k > 1:
            (i, j), k = max_square(M)
            f.write(fill(i-k+1, j-k+1, k))            
            M[i-k+1:i+1, j-k+1:j+1] = np.zeros((k, k))
    #    e = 2
    #    while e > 1:
    #        S = sum_mat(M)
    #        e, (a, b) = best_ksquare(S, 2)
    #        fill_erase(f, M, a, b, 2)
        for i in range(len(M)):
            for j in range(len(M[i])):
                if M[i, j] == 1: 
                    f.write(fill(i, j, 1))
```

```python
greedy(im)
```

```python

def erase(i, j):
    return "ERASE," + str(j) + "," + str(i) + "\n"

def squarify(M, erases, p):
    change = True
    while change:
        change = False
        for i in range(len(M)):
            for j in range(len(M[i])):
                if M[i, j] == 1: continue
                voisins = 0
                for a in [i-1, i, i+1]: #[(i-1, j), (i, j-1), (i+1, j), (i, j+1), ]:
                    for b in [j-1, j, j+1]:
                        if (a,b) != (i,j) and 0<=a<len(M) and 0<=b<len(M[0]) and M[a, b] == 1:
                            voisins += 1
                if voisins > p: 
                    change = True
                    M[i, j] = 1
                    erases.append(erase(i, j))
```

```python
def greedy(M):
    f = open("operations.txt", "w")
    sz = 2
    erases = []
#    squarify(M, erases)
    k = len(erases)
    print(k)
    while sz > 1:
        (i, j), sz = max_square(M)
        f.write(fill(i, j, sz))            
        M[i-sz+1:i+1, j-sz+1:j+1] = np.zeros((sz, sz))
        if k % 100 == 0: print(k, sz)
        k += 1
    for i in range(len(M)):
        for j in range(len(M[i])):
            if M[i, j] == 1: 
                k += 1
                f.write(fill(i, j, 1))
    print(k)
    f.writelines(erases)
    f.close()
    
greedy(im)
```

```python
def fill(i, j, k):
    return "FILL," + str(j) + "," + str(i) + "," + str(k) + "\n"
                         
def erase(i, j):
    return "ERASE," + str(j) + "," + str(i) + "\n"

def sum_mat(M):
    S = np.zeros((len(M)+1, len(M[0])+1), dtype = np.int64)
    for i in range(1, len(S)):
        for j in range(1, len(S[0])):
            S[i, j] = S[i-1, j] + S[i, j-1] - S[i-1, j-1] + M[i-1, j-1]
    return S

def best_ksquare(S, k):
    D1 = S[k:, :] - S[:-k, :]
    D2 = D1[:, k:] - D1[:, :-k]
    pos = np.unravel_index(np.argmax(D2, axis=None), D2.shape)
    eff = D2[pos]/(1 + k*k - D2[pos])
    return eff, pos

def fill_M(f, M, i, j, k):
    f.write(fill(i, j, k)) 
    M[i:i+k, j:j+k] = np.ones((k, k), dtype = np.int64)
    
def greedy(im):
    M = np.zeros_like(im)
    f = open("operations.txt", "w")
    k = 85
    while k > 1:
        S = sum_mat(im > M)
        e, (a, b) = best_ksquare(S, k)
        if k == 2: eff2 = 1
        else: eff2, (_, _) = best_ksquare(S, k-1)
        if eff2 > e: 
            k -= 1
        else:
            fill_M(f, M, a, b, k)
            print(k, e)
    for i in range(len(M)):
        for j in range(len(M[i])):
            if M[i, j] < im[i, j]: 
                f.write(fill(i, j, 1))
            if M[i, j] > im[i, j]: 
                f.write(erase(i, j))
    f.close()
    return M
    
#greedy(turing("turing.txt"))
```

```python

M = np.round(np.random.rand(6,8) > 0.2)
print(M)
greedy(M)
print(M)
```
```python
S = sum_mat(M)
D1 = S[k:, :] - S[:-k, :]
D2 = D1[:, k:] - D1[:, :-k]
e, (a, b) = best_ksquare(S, k)
f = open("operations.txt", "w")
#fill_erase(f, M, a, b, k)

f.close()
```

```python

def squarify(im, erases, p):
    change = True
    while change:
        change = False
        for i in range(len(im)):
            for j in range(len(im[i])):
                if im[i, j] == 1: continue
                voisins = 0
                for a in [i-1, i, i+1]: 
                    for b in [j-1, j, j+1]:
                        if (a,b) != (i,j) and 0<=a<len(im) and 0<=b<len(im[0]) and im[a, b] == 1:
                            voisins += 1
                if voisins > p: 
                    change = True
                    im[i, j] = 1
                    erases.append(erase(i, j))
                    
def best_ksquare(S, k):
    D1 = S[k:, :] - S[:-k, :]
    D2 = D1[:, k:] - D1[:, :-k]
    pos = np.unravel_index(np.argmax(D2, axis=None), D2.shape)
    eff = D2[pos]/(1 + k*k - D2[pos])
    return eff, pos

def fill_M(f, M, i, j, k):
    f.write(fill(i, j, k)) 
    M[i:i+k, j:j+k] = np.ones((k, k), dtype = np.int64)
    
def best_square(S, kinit):
    maxi, a, b, kmax = 0, 0, 0, 0
    for k in range(kinit, -1, -1):
        if maxi >= k*k:
            #if maxi != kmax*kmax:
            print(maxi, kmax)
            return a, b, kmax
        e, (i, j) = best_ksquare(S, k)
        if e > maxi:
            maxi, a, b, kmax = e, i, j, k
    
def greedy(im):
    M = np.zeros_like(im)
    f = open("operations.txt", "w")
    k = 90
    while True:
        S = sum_mat(im > M)
        a, b, k = best_square(S, 90)
        if k <= 1: break
        fill_M(f, M, a, b, k)
    for i in range(len(M)):
        for j in range(len(M[i])):
            if M[i, j] < im[i, j]: 
                f.write(fill(i, j, 1))
            if M[i, j] > im[i, j]: 
                f.write(erase(i, j))
    f.close()
    return M
    
greedy(turing("turing.txt"))
```
