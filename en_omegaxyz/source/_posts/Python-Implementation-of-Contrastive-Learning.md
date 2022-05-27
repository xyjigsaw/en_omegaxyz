---
title: Python Implementation of Contrastive Learning
date: 2022-01-13 16:40:25
tags: [Python, machine learning]
categories: technology
math: true
index_img: https://raw.githubusercontent.com/xyjigsaw/image/master/upload/contrastive_learning202201131713.png
---

## Introduction
Contrastive learning is a self-supervised learning method to learn representations by contrasting positive and negative examples. For self-supervised contrastive learning, the next equation shows the contrastive loss:

$$\mathcal{L}_{i,j} = - \log \frac{exp(\textbf{z}_i \cdot \textbf{z}_j / \tau)}{\sum_{k=1,k\neq i}^{2N}exp(\textbf{z}_i \cdot \textbf{z}_k / \tau)},$$
where $$\textbf{z}_i$$ is the embedding of sample $$i$$ and $$\tau \in \mathcal{R}^{+}$$ is the temperature parameter.

## Codes

There are two versions to implement contrastive loss：

### Only augment positive data such as graph classification

```python
def contrastive_loss(x, x_aug, T):
    """
    :param x: the hidden vectors of original data
    :param x_aug: the positive vector of the auged data
    :param T: temperature
    :return: loss
    """
    batch_size, _ = x.size()
    x_abs = x.norm(dim=1)
    x_aug_abs = x_aug.norm(dim=1)

    sim_matrix = torch.einsum('ik,jk->ij', x, x_aug) / torch.einsum('i,j->ij', x_abs, x_aug_abs)
    sim_matrix = torch.exp(sim_matrix / T)
    pos_sim = sim_matrix[range(batch_size), range(batch_size)]
    loss = pos_sim / (sim_matrix.sum(dim=1) - pos_sim)
    loss = - torch.log(loss).mean()
    return loss
```


### augment positive and negative sample


```python
def info_nce_loss(self, features):
    labels = torch.cat([torch.arange(self.args.batch_size) for i in range(self.args.n_views)], dim=0)
    labels = (labels.unsqueeze(0) == labels.unsqueeze(1)).float()
    labels = labels.to(self.args.device)

    features = F.normalize(features, dim=1)

    similarity_matrix = torch.matmul(features, features.T)
    # assert similarity_matrix.shape == (
    #     self.args.n_views * self.args.batch_size, self.args.n_views * self.args.batch_size)
    # assert similarity_matrix.shape == labels.shape

    # discard the main diagonal from both: labels and similarities matrix
    mask = torch.eye(labels.shape[0], dtype=torch.bool).to(self.args.device)
    labels = labels[~mask].view(labels.shape[0], -1)
    similarity_matrix = similarity_matrix[~mask].view(similarity_matrix.shape[0], -1)
    # assert similarity_matrix.shape == labels.shape

    # select and combine multiple positives
    positives = similarity_matrix[labels.bool()].view(labels.shape[0], -1)

    # select only the negatives the negatives
    negatives = similarity_matrix[~labels.bool()].view(similarity_matrix.shape[0], -1)

    logits = torch.cat([positives, negatives], dim=1)
    labels = torch.zeros(logits.shape[0], dtype=torch.long).to(self.args.device)

    logits = logits / self.args.temperature
    return logits, labels

self.criterion = torch.nn.CrossEntropyLoss()
loss = self.criterion(logits, labels)
```

OmegaXYZ.com
All rights reserved.


