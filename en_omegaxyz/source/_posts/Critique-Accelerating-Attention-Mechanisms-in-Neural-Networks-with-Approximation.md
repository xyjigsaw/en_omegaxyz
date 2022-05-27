---
title: 'Critique: Accelerating Attention Mechanisms in Neural Networks with Approximation'
date: 2021-01-20 22:38:40
tags: [computer architecture,paper]
categories: technology
index_img: https://raw.githubusercontent.com/xyjigsaw/image/master/upload/aaa_nn1202101202239.jpg
---

> **Paper**: Ham, Tae Jun, et al. "A^ 3: Accelerating Attention Mechanisms in Neural Networks with Approximation." 2020 IEEE International Symposium on High Performance Computer Architecture (HPCA). IEEE, 2020.

[Paper PDF](https://ieeexplore.ieee.org/abstract/document/9065498/)

## Summary

The paper designs a specialized hardware accelerator called A3 that targets attention mechanisms in neural networks where approximation potential is exploited. In particular, the work of A3 identiﬁes the importance of the emerging neural network primitive and accelerates it with software-hardware co-design to achieve orders of magnitude energy efﬁciency improvement over the conventional hardware. Besides, A3 architects the specialized hardware pipeline for vanilla and approximate attention mechanism while a test chip at TSMC 40nm is taped out. Experimental results have demonstrated that the accelerator achieves significant performance and energy efficiency gains over conventional hardware.

This paper is well written and organized. It contains four main contributions including a novel approximation algorithm, a specialized hardware pipeline, better performance, and efficiency. To be specific, it states and analyzes specific problems in attention mechanism, proposes solutions based on hardware accelerator, and gives convincing experiments to evaluate the pros and cons of solutions. The proposed approximate accelerator is exquisitely designed except for its applicability and commercial value which implies the technique here may well not keep up with the diversity of different neural networks. Overall, this paper is worth reading for AI researchers. I have a positive view of it.

In the next few sections, I will discuss this paper from several aspects including the creativity of the research, the way to solve the problem, experimental design, and applicability.

## Motivation and Creativity

In section Ⅱ-A and Ⅱ-B, it shows that attention mechanism is a widely used strategy in most state-of-the-art neural networks such as Word2Vec, Glove, and FastText to identify and retrieve data relevant to the input which is differentiable content-based similarity search. Most of the networks are in the field of natural language processing, computer vision, and recommendation system. The paper analyzes the computational process of the dot product, softmax normalization, and weight sum during the attention mechanism in detail. Afterward, the paper draws the conclusion that most of the computations performed in the matrix-vector multiplication have very little impact on the ﬁnal output since most score values become near-zero after the softmax normalization which can be approximated and optimized. Thus, the A3 accelerator is around the corner to be introduced in the paper. 

From my perspective, the motivation in the paper is enlightened from Amdahl’s Law [1] one of which is the importance to accelerate relatively less optimized portion. Therefore, the neural network primitives should be optimized and there exists a lot of research in this field. In the open literature, it is the first time [2] the attention mechanism is considered in the level of accelerator hardware with approximation which shows the superiority of the paper to a certain extent. Obviously, the paper provides readers a detailed explanation of the motivation such as the pseudocode (figure 1.) and the example application of attention mechanism (figure 2.) to devise such accelerator. Readers can easily understand how the following A3 accelerator is designed and reduces computation dramatically. 

## Solutions

This paper introduces two different versions of A3: Base-A3 (section III) and Approx-A3 (section IV and V). Base-A3 is for base attention while Approx-A3 is for approximate attention. For the former, each module’s hardware design directly maps to its computation. Therefore, the latter is more worthy of discussion since the approximate mechanism is proposed here. 

Particularly, ideas on how to devise approximate attention have two critical steps. One is the identification of candidates that are relevant to the query in the attention mechanism with limited computation. The other is to avoid computation for likely to be non-relevant rows. There is a key intuition: can compute the estimated attention score with little computation if we can somehow identify a few largest and a few smallest component multiplication results.

For Approx-A3, the authors design a new set of hardware accelerator modules for candidate selection and post-scoring approximation. It uses the naïve idea that addition is better than multiplication. For example, given a matrix of size n by d, Approx-A3 sorts each column of the matrix stored in SRAM firstly. Then two pointers of size 1 by d aim to fetch the max and min elements in the sorted column for m times to update the estimated attention in place of the element-wise multiplication for query vector and sorted matrix. Thus, the algorithm only performs 2 by m multiplications, which is much smaller than n by d. In a word, the algorithm updates two estimated attention scores per iteration: largest and smallest component multiplication results. Finally, rows with positive estimated attention scores after m iterations become candidates for approximate attention.

![Chip](https://raw.githubusercontent.com/xyjigsaw/image/master/upload/aaa_nn2202101202246.jpg)

I have noticed that there is a sorting operation in the process when approximation and I thought it would affect the performance of A3 at the beginning. The sorting operation (O(logn)) will take up additional computing resources and authors should carefully adopt sorting. Fortunately, authors have considered the performance overhead of sort. In section IV, the paper shows sorting may not be on the critical path, and the same key matrix is re-used for multiple queries for workloads with the self-attention mechanism which can be amortized over multiple queries. In addition, for memory network models, information can be preprocessed before the query arrives. Sorting does not affect query time. In this part, I hope that the paper could have further argumentation and analysis in detail. We can conclude that the design of A3 obeys the principle of locality which is the reuse of data and instructions.

For the issue of test chip, I find that authors used n = 320 and d = 64 to ﬁt their largest workload in section VI-D. In most neural networks, the problem dimension (d) can easily exceed 64. I wonder whether such an accelerator can handle high-dimensional attention mechanism. In other words, for query vector with more than 64 dimensions (d > 64), does it need to design related methods at the hardware level? The question is not answered in the paper.

In my opinion, the approximate attention is well designed and hardware-friendly. On the one hand, it reduces a lot of unnecessary calculations with high efficiency. On the other hand, it also satisfies the demand of chip design where query vector register, sorted key matrix SRAM, pointers, multipliers, and component multiplication buffer are needed for full-pipelining. Approx-A3 sets the stage for significantly improving the performance of neural network computation for attention mechanism.

## Experiments and Evaluation
The paper demonstrates several experiments to evaluate the A3 accelerator. Selection VI is divided into four parts: A (Workloads), B (Accuracy Evaluation), C (Performance Results) and D (Area, Power, Energy and Test Chip).

From the performance results, it can be seen that Approximation enables even further throughput improvement (2.6-7.0×) as well as latency improvement (1.6-8.0×). Thus, when it comes to area and energy efficiency, more energy can be saved (>10,000× more efficient than CPU). The result proves that the previous design of Approx-A3 is very effective. This is useful if such a technique is applied to the mobile terminal under the circumstance that the die size is ignored. Furthermore, it should be noted that most energy is spent on output computation and candidate selection which can be understood easily since the element-wise multiplication is replaced by approximation.

However, we all know that the approximation scheme affects end-to-end model accuracy. From VI-B, the results show that the conservative approximation scheme loses about 1-1.6% accuracy metric while the aggressive approximation scheme loses about 8-9% accuracy metric. Additionally, the amount of top entries selected shows that the aggressive approximation scheme may miss some items with high attention scores.

How to define conservative and aggressive is a challenge for users though the result looks pretty good and convincing. The paper says there are two configurable parameters (M: iteration count for candidate selection algorithm, T: the threshold for post-scoring selection algorithm) to balance the efficiency and accuracy. Naturally, the larger M, the accuracy of the model increases and the lower T indicates more conservative approximation while the higher T indicates more aggressive approximation. However, it will lose beneﬁts of approximation with a large number of candidates and affect the accuracy with higher T value. How to choose appropriate M and T for users? This question is still ambiguous to readers and has not been discussed in depth. As far as I’m concerned, the influence on the training of the upstream neural network is still debatable. Moreover, experiments on accuracy performance for approximation scheme should choose more datasets and attention based state-of-the-art algorithms to prove its generalization.

## Others

The idea in this paper is simple but effective for attention mechanism. Nevertheless, I hold the view that it is very difficult for A3 accelerator to make commercial applications. Then, I will state three intuitive reasons. Firstly, neural networks are changing and updating with each passing day, attention mechanism is only a small part of it and someday in the future, it will become obsolete definitely. I prefer to analyze more general neural network components such as Transformer for accelerator rather than a specific domain. Secondly, the cost of deploying A3 accelerators will become very high, not to mention the reduction of die size. Last but not least, one of the contributions in this paper is approximate attention algorithm, maybe we can optimize attention mechanism at the software level instead of the hardware level further. 

Thus, it would be more interesting if the author could introduce their future work. Of course, the solution to reduce computations in this paper is worth learning and it is of great help to the design of general hardware component for neural networks in the future.

## References

[1] Gustafson, John L. "Reevaluating Amdahl's law." Communications of the ACM 31.5 (1988): 532-533.
[2] Lu, Siyuan, et al. "Hardware Accelerator for Multi-Head Attention and Position-Wise Feed-Forward in the Transformer." arXiv preprint arXiv:2009.08605 (2020).


> [Original](https://www.omegaxyz.com/2020/10/16/a3-hpca2020/) 
