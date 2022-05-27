---
title: 'von Neumann Graph Entropy: Python Implementation'
date: 2021-01-25 23:50:04
tags: [Python, machine learning]
categories: technology
math: true
index_img: https://raw.githubusercontent.com/xyjigsaw/image/master/upload/social-network202101260047.jpg
---

## Introduction

The von Neumann graph entropy (VNGE) facilitates measurement of information divergence and distance between graphs in a graph sequence. 

Given an undirected graph $$G=(V, E, A)$$, where $$A$$ is the symmetric weight matrix. The degree matrix is defined as $$D=diag(d_1,...,d_n)$$ and the Laplacian matrix is defined as $$L=D-A$$, its eigenvalues $$\lambda_i$$ are called Laplacian spectrum. Here, $$H_{vn}(G)$$ is called von Neumann graph entropy.

$$H_{vn}(G)=-\sum \limits_{i=1}^{n}(\frac{\lambda_i}{vol(G)}\log\frac{\lambda_i}{vol(G)})$$

where the volume of a graph is $$vol(G)=\sum_{i=1}^{n}\lambda_i=trace(L)$$. The time complexity of calculating von Neumann graph entropy is $$O(n^3)$$.

I provide a Python version code to calculate VNGE. It should be noted that Chen. et al give an approximate method called FINGER[1] which reduces the cubic complexity of VNGE to linear complexity in the number of nodes and edges. 

TheCodes of VNGE and FINGER are as follows.

## Codes

```python
# Name: VNGE
# Author: Reacubeth
# Time: 2021/1/25 16:01
# Mail: noverfitting@gmail.com
# Site: www.omegaxyz.com
# *_*coding:utf-8 *_*

import time
import numpy as np
from scipy.sparse.linalg.eigen.arpack import eigsh


def normalized_laplacian(adj_matrix):
    nodes_degree = np.sum(adj_matrix, axis=1)
    nodes_degree_sqrt = 1/np.sqrt(nodes_degree)
    degree_matrix = np.diag(nodes_degree_sqrt)
    eye_matrix = np.eye(adj_matrix.shape[0])
    return eye_matrix - degree_matrix * adj_matrix * degree_matrix


def unnormalized_laplacian(adj_matrix):
    nodes_degree = np.sum(adj_matrix, axis=1)
    degree_matrix = np.diag(nodes_degree)
    return degree_matrix - adj_matrix


def VNGE_exact(adj_matrix):
    start = time.time()
    nodes_degree = np.sum(adj_matrix, axis=1)
    c = 1.0 / np.sum(nodes_degree)
    laplacian_matrix = c * unnormalized_laplacian(adj_matrix)
    eigenvalues, _ = np.linalg.eig(laplacian_matrix)
    eigenvalues[eigenvalues < 0] = 0
    pos = eigenvalues > 0
    H_vn = - np.sum(eigenvalues[pos] * np.log2(eigenvalues[pos]))
    print('H_vn exact:', H_vn)
    print('Time:', time.time() - start)


def VNGE_FINGER(adj_matrix):
    start = time.time()
    nodes_degree = np.sum(adj_matrix, axis=1)
    c = 1.0 / np.sum(nodes_degree)
    edge_weights = 1.0 * adj_matrix[np.nonzero(adj_matrix)]
    approx = 1.0 - np.square(c) * (np.sum(np.square(nodes_degree)) + np.sum(np.square(edge_weights)))
    laplacian_matrix = unnormalized_laplacian(adj_matrix)
    '''
    eigenvalues, _ = np.linalg.eig(laplacian_matrix)  # the biggest reduction
    eig_max = c * max(eigenvalues)
    '''
    eig_max, _ = eigsh(laplacian_matrix, 1, which='LM')
    eig_max = eig_max[0] * c
    H_vn = - approx * np.log2(eig_max)
    print('H_vn approx:', H_vn)
    print('Time:', time.time() - start)


nodes_num = 3000
sparsity = 0.01


tmp_m = np.random.uniform(0, 1, (nodes_num, nodes_num))
pos1 = tmp_m > sparsity
pos2 = tmp_m <= sparsity
tmp_m[pos1] = 0
tmp_m[pos2] = 1
tmp_m = np.triu(tmp_m)
adj_m = tmp_m + np.transpose(tmp_m)

VNGE_exact(adj_m)
VNGE_FINGER(adj_m)
```

## Results

```bash
H_vn exact: 11.496599468152386
Time: 13.172455072402954
H_vn approx: 10.63029083591871
Time: 0.23734617233276367
```

[1] Chen, Pin-Yu, et al. "Fast incremental von neumann graph entropy computation: Theory, algorithm, and applications." International Conference on Machine Learning. PMLR, 2019.

OmegaXYZ.com
All rights reserved.


