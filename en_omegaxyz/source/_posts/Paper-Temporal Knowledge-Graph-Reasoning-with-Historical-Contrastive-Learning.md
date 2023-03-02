---
title: 'Paper: Temporal Knowledge Graph Reasoning with Historical Contrastive Learning'
date: 2022-12-03 11:28:11
tags: [Python, machine learning, knowledge graph]
categories: technology
math: true
index_img: https://raw.githubusercontent.com/xyjigsaw/image/master/upload/cenet_tkg_2023_03_02_10.png
---


## Temporal Knowledge Graph


Knowledge Graphs (KGs), as a collection of human knowledge, have shown great prospects in natural language processing, recommendation systems and information retrieval. The traditional KG is usually a static knowledge base. It uses a graph structure data topology and integrates facts (also known as events) in the form of triples $$(s, p, o)$$, where $$s$$ and $$o$$ represent subject (header entity) and object (tail entity) entities respectively, and $$p$$ is used as a relationship type to express predicates. In the real world, due to the continuous development of knowledge, the construction and application of temporal knowledge atlas (TKG) has become a hot topic in the field, in which triples $$(s, p, o)$$ are expanded to quintuples, and time stamps $$t$$, namely $$(s, p, o, t)$$, are added. The following figure shows a TKG composed of a series of international political events.

![A Temporal Knowledge Subgraph](https://raw.githubusercontent.com/xyjigsaw/image/master/upload/cenet_tkg_2023_03_02_10.png)


## Temporal Knowledge Graph Reasoning

TKG has provided new perspectives and insights for many downstream applications, such as decision-making, stock forecasting and dialogue systems, thus arousing people's strong interest in TKG reasoning. In this work, we focus on predicting future events on TKG.

In order to model the structure and time characteristics of TKG for future event prediction, some mainstream models, such as RE-NET, can easily predict repeated or periodic events using autoregressive methods. However, in the TKG dataset ICEWS (Integrated Crisis Early Warning System), new events account for about 40%.

It is very challenging to predict these new events because they have fewer traces of interaction on the historical timeline. For example, the right part of the following figure shows the query (the United States, Negotiate,?, $$t+1$$) and its corresponding new events (the United States, Negotiate, Russia, $$t+1$$). Most of the existing methods usually get wrong results in this type of query because they pay too much attention to frequent repeated events.

![Problems in Existing Models](https://raw.githubusercontent.com/xyjigsaw/image/master/upload/cenet_problems_2023_03_02_10.png)


On the other hand, in the reasoning process, the existing methods rank the probability scores of all candidate entities in the whole graph without any bias. We believe that this kind of bias is very necessary when dealing with missing entities of different events, which is also true in real society. For example, for repetitive or periodic events, we should give priority to some entities that occur frequently, while for new events, we need to pay more attention to entities with less historical interaction.

## Contrastive Event Network (CENET)

In this work, we will go beyond the limits of historical information and explore potential temporal patterns from the whole knowledge. In order to clarify our design more clearly, we call the past events associated with the entities in the current query historical events of the query, and other events non-historical events. We intuitively believe that the events in TKG are not only related to their historical events, but also indirectly related to the potential factors that have not been observed. The historical events we can see are only the tip of the iceberg. We propose a new TKG reasoning model called CENET (Contrastive Event Network). The following figure shows the framework of the model.

![The overall architecture of CENET. The left part learns the distribution of entities from both historical and non-historical dependency. The right part illustrates the two stages of historical contrastive learning, which aims to identify highly correlated entities, and the output is a boolean mask vector. The middle part is the mask-based inference process that combines the distribution learned from the two kinds of dependency and the mask vector to generate the final results.](https://raw.githubusercontent.com/xyjigsaw/image/master/upload/cenet_architecture_2023_03_02_10.png)


Given the query $$(s, p, ?, t)$$ whose real tail entity is $$o$$, CENET considers its historical and non-historical events, and identifies important entities through contrastive learning. Specifically, the model first uses a scoring strategy based on copy-mechanism to model the correlation between historical events and non-historical events.

$$\mathbf{H}_{his}^{s,p}=\underbrace{tanh(\mathbf{W}_{his}(\mathbf{s}\oplus \mathbf{p}) + \mathbf{b}_{his})\mathbf{E}^{T}}_{\text{similarity score between $q$ and $\mathcal{E}$}} + \mathbf{Z}_{t}^{s,p}$$

$$\mathbf{H}_{nhis}^{s,p}=tanh(\mathbf{W}_{nhis}(\mathbf{s}\oplus \mathbf{p}) + \mathbf{b}_{nhis})\mathbf{E}^{T} - \mathbf{Z}_{t}^{s,p}$$

In addition, all queries can be divided into two categories according to their real object entities: tail entities are historical entities or non-historical entities. Therefore, CENET naturally uses supervised contrastive learning loss to train the representation of two types of queries (i.e., $$v_q $$in formula 3), further helping to train the classifier whose output is Boolean, so as to identify which entities should receive more attention. In the process of reasoning, CENET combines the distribution of historical and non-historical dependencies, and further uses the mask-based strategy to consider highly relevant entities according to the classification results.

$$\mathcal{L}^{sup} = \sum_{q \in M}\frac{-1}{|Q(q)|}\sum_{k \in Q(q)}\log \frac{exp(\mathbf{v}_q \cdot \mathbf{v}_k / \tau)}{\sum\limits_{a \in M \backslash \{q\}}(\mathbf{v}_q \cdot \mathbf{v}_a / \tau)}$$


![Historical Comparative Learning: CENET uses contrastive loss in the first stage, and uses cross entropy loss to train binary classifier in the second stage.](https://raw.githubusercontent.com/xyjigsaw/image/master/upload/cenet_contrastive_2023_03_02_10.png)

## Experiments

### Datasets

We selected five benchmark data sets, including three event-based TKGs and two public KG. These two types of data sets are constructed in different ways. The first three TKGs based on international political events are composed of ICEWS18, ICEWS14 and GDELT, and the events are discrete. Events in the last two public KG (WIKI and YAGO) may exist continuously for a long time.

![Datasets](https://raw.githubusercontent.com/xyjigsaw/image/master/upload/cenet_datasets_2023_03_02_10.png)

### Comparative Experiment

We selected 15 of the industry's latest TKG reasoning models as the baseline, and adopted Mean Rational Ranks (MRR) and Hits@1/3/10 (the proportion of correct judgments ranked within top 1/3/10) as the evaluation index. The following table shows the results. The results show that CENET is significantly superior to all existing methods in most metrics. Compared with the most advanced baseline before, Hits@1 At least 8.3% relative improvement was achieved.

![Results of event-based TKG](https://raw.githubusercontent.com/xyjigsaw/image/master/upload/cenet_cmp_2023_03_02_10.png)

![Results of public KG](https://raw.githubusercontent.com/xyjigsaw/image/master/upload/cenet_cmp2_2023_03_02_10.png)

### Ablation Experiment

We analyzed the performance of each key component in the model, and the results of the ablation experiment are shown in the following table. It can be seen that taking historical and non-historical dependencies into account at the same time can achieve better results than only taking historical dependencies into account. The historical contrastive learning strategy and mask-based inference can achieve superior performance.

![Ablation Results](https://raw.githubusercontent.com/xyjigsaw/image/master/upload/cenet_ablation_2023_03_02_10.png)


### Case Study

The case analysis shows the case of the model in predicting repetitive events and new events.
![Case Analysis](https://raw.githubusercontent.com/xyjigsaw/image/master/upload/cenet_case_study_2023_03_02_10.png)

## Conclusion
In this paper, we propose a new temporal knowledge graph model for event prediction: contrastive event network (CENET). The key idea of CENET is to learn the convincing distribution of the entire entity set and identify important entities from historical and non-historical dependencies in the contrastive learning framework. The experimental results show that CENET is significantly superior to all existing methods in most metrics, especially in Hits@1. Future work includes exploring the contrastive learning ability in the knowledge graph, such as finding more reasonable contrastive pairs.

The work has been accepted by AAAI 2023. The paper and code link are as follows:
**Paper Link:**
[https://arxiv.org/abs/2211.10904](https://arxiv.org/abs/2211.10904)

**Code Link:**
[https://github.com/xyjigsaw/CENET](https://github.com/xyjigsaw/CENET)

OmegaXYZ.com
All rights reserved.
