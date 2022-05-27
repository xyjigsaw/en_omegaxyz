---
title: 'Critique: DNNGuard: An Elastic Heterogeneous DNN Accelerator Architecture against Adversarial Attacks'
date: 2021-04-12 21:07:59
tags: [computer architecture,paper]
categories: technology
index_img: https://raw.githubusercontent.com/xyjigsaw/image/master/upload/DNNGuard202104122126.jpeg
---


>**Paper**: Wang, Xingbin, et al. “Dnnguard: An elastic heterogeneous dnn accelerator architecture against adversarial attacks.” Proceedings of the Twenty-Fifth International Conference on Architectural Support for Programming Languages and Operating Systems. 2020.

[Paper PDF](https://dl.acm.org/doi/abs/10.1145/3373376.3378532)

From my perspective, it is a comprehensive paper that combines computer architecture, security, privacy, and artificial intelligence. My research interest is knowledge graph and social network, but I think this paper is suitable for every researcher in all fields of computer science and I hold a positive view of the proposed sophisticated architecture. Next, I will briefly introduce and analyze the work to better share the ideas.

## Summary

Should we believe in machine learning and artificial intelligence? The work in this paper tries to solve the frequently occurring security problems in DNN which is a novel and emerging direction in computer science. Nowadays, the deep neural network is vulnerable to adversarial samples, but the existing DNN accelerators have many problems in the detection of adversarial samples’ attack, let alone memory cost, computational efficiency, and information security. Moreover, these mentioned accelerators do not provide effective support for the special calculations required for detection methods. These are the basic problems to be solved in this paper.

This paper proposes an elastic heterogeneous DNN accelerator architecture called DNNGuard which consists of three key parts: 1) an elastic on-chip buffer management mechanism that can fully exploit the data locality, 2) an elastic PE computing resource management which makes the detection network execute faster than target network to avoid false prediction by identifying possible attacks timely while maximizing the utilization of computing resources 3) an extended AI instruction set to support the synchronization and communication mechanisms. All these parts have been carefully designed and experimentally verified.

As I have mentioned above, in general, this paper focuses on the challenges of DNN security where injected malicious data like adversarial samples will cause disastrous consequences. As usual, the clarity, novelty, and technical correctness of the paper will be discussed in detail fairly.

![](https://raw.githubusercontent.com/xyjigsaw/image/master/upload/DNNGuard202104122126.jpeg)

## Clarity & Organization

Generally speaking, the organization of this paper is clearly structured. Firstly, the authors give a whole overview of the introduction part. In order to illustrate readers to better understand the content of the article, the paper analyzes the existing adversarial sample defense methods and the requirements for accelerator architecture in Section 2. In Section 3, the framework of DNNGuard is presented with implementation details explained in Section 4. Like most papers, the authors evaluate the performance impacts and parameter sensitivity for DNNGuard in Section 5 and discuss various design issues in Section 6. At the end of the paper, the related work is reviewed in Section 7 before concluding the paper in Section 8. 


However, as far as I am concerned, I think the abstract part is slightly longer which covers the entire first page with other basic information. Specifically, the background in the abstract has nearly 100 words which look redundant. It is supposed to summarize the related preliminaries of the urgent security situation in two or three sentences while expressing the richest meaning with minimum words for the key ideas and methods of DNNGuard. If there is really something important related to the architecture that is not described, the authors can describe it in the main text.

What needs to be pointed out is that there are many concepts, terminologies, and models in this paper such as Scheduler, PE, MAC, and CACC. The paper could be more reader-friendly to list these words like paper [1] did. 

Overall, this part is generally satisfactory in terms of clarity and organization.

## Novelty

Novelty is not only the core of all papers but also the focus of this critique. There is no doubt that this article has many innovations. 

First, in Section 3 and 4, the authors consider that there are two different networks (target network and detect network) need to be executed in the situation where adversary samples exist. In the realm of my knowledge, no accelerator architecture that can convert the serial execution of two networks into parallel execution. Therefore, I think the utilization of parallelism is the most significant contribution to this work. Since simply reusing the DNN accelerator does not achieve the highest efficiency, the authors thought of utilizing the CPU to do some serial tasks to ease the pressure on the DNN accelerator, but this will bring high latency caused by data movement. Moreover, due to the deterministic and sequential nature of the target network and the detection network’s processing flow, the authors do not use a complex handshake mechanism to synchronize and schedule the tasks. Instead, a scheduler within the DNN accelerator with an extended AI instruction set is creatively introduced in this paper to dynamically configure the PE and on-chip buffer resources. The author's ability to find innovative solutions based on existing problems is worth learning. 

What’s more, we can see the architecture in Figure 2 (The DNNGuard Architecture Based on Elastic DNN Accelerator and CPU Core). Specially, we can find that the CPU core is mainly used for executing the special computing units while the elastic DNN accelerator core is composed of the scheduler, Soc Bus Interface, and the global buffers. One of the biggest advantages and innovations of DNNGuard is the global buffer which is the key to provide efficient data communication in such a complex architecture. Additionally, the authors have designed a complex storage model and corresponding data movement mechanism to effectively calculate these two neural networks simultaneously.

In the discussion section (Section 6), the paper provides several interesting angles to totally analyze the architecture besides performance and effectiveness such as 1) the robust target network, 2) the adaptability to future algorithms, 3) the compatibility with the current DNN accelerator, and 4) the strong security. These aspects are also the critical ingredients of the proposed architecture.

At the end of this part, I have two questions as follows:
- Regarding the training of the two networks, can the training of the detection network be combined with the training of the target network? That is to say, it is possible to integrate these two networks together furtherly considering that the target network is based on traditional DNN or other deep learning models and the detection network may be related to DNN or machine learning. 
- Are there some new problems coming up the integrated design of DNNGuard architecture brings? It is an open problem. I think there is room for further improvement.

## Technical Correctness and Integrity of Experiments

In this part, the experimental parts will be analyzed in detail where NVDLA[2] and RISCV[3] are employed to implement the accelerator of DNNGuard. 

As a new orchestrated architecture, it is supposed to evaluate the performance on different networks, the area and power it consumes compared to other existing architecture, and sensitivity related to the ratio of frame or bandwidth, etc. Fortunately, the experiments in the work demonstrate kinds of comprehensive evaluations. In addition to what I mentioned above, the work also compares the proposed architecture with others using Non-DNN Defense Methods on CPU. In the part of sensitivity analysis, the authors analyze the impacts of the number of PE, the buffer capacity, DRAM bandwidth, and LLC size of CPU which are the main factors to the whole performance.

The paper lists many results in different experimental parts. Section 5.2 (Results on DNNGuard Architecture) and 5.3 (Detection Mechanism on DNNGuard) are the most important parts where the outcomes of the architecture prove the superiority of DNNGuard over other existing architecture in terms of the elastic NVDLA performance and DRAN access.

However, the paper mentioned the protection for information security and privacy in neural networks. According to their theory, tightly coupling the DNN accelerator and the CPU core in a chip can effectively avoid side-channel information leakage of data interface compared with the deployment scheme of connecting two DNN accelerators through PCIe interface. Nevertheless, the whole paper only mentions the fact rather than theoretical analysis in Section 6. I still do not know why this can improve the security of information due to the limitations of my knowledge. The paper is supposed to leverage existing attack models to test and evaluate the security level of DNNGuard. Only in this way can readers be convinced.

## Conclusion

In summary, this paper is an excellent work with significant contributions in the aspects of neural network architecture against adversarial attacks though there are a few unsatisfactory but trivial in the experimental part. 


## References

[1] Wu, Zonghan, et al. "A comprehensive survey on graph neural networks." IEEE Transactions on Neural Networks and Learning Systems (2020).
[2] NVIDIA. Hardware architectural specification. http://nvdla.org/hw/v1/hwarch.html, 2018.
[3] Andrew Waterman, Yunsup Lee, David A Patterson, and Krste Asanovi. The risc-v instruction set manual. volume 1: User-level isa, version 2.0. Technical report, CALIFORNIA UNIV BERKELEY DEPT OF ELECTRICAL ENGINEERING AND COMPUTER SCIENCES, 2014.


> [Original](https://www.omegaxyz.com/2021/03/31/dnn-guard/) 
