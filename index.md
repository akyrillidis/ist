---
layout: default
---

Neural networks (NN) are ubiquitous, and have led to the recent success of machine learning (ML). For many applications, model training and inference need to be carried out in a timely fashion on a computation-/communication-light, and energy-limited platform. Optimizing the NN training in terms of computation/com- munication/energy efficiency, while retaining competitive accuracy, has become a fundamental challenge.

We propose a new class of neural network (NN) training algorithms, called independent subetwork training (IST). IST decomposes the NN into a set of independent subnetworks. Each of those subnetworks is trained at a different device, for one or more backpropagation steps, before a synchronization step. Updated subnetworks are sent from edge-devices to the parameter server for reassembly into the original NN, before the next round of decomposition and local training. Because the subnetworks share no parameters, synchronization requires no aggregation; it is just an exchange of parameters. Moreover, each of the subnetworks is a fully-operational classifier by itself; no synchronization pipelines between subnetworks are required. Key attributes of our proposal is that $$i)$$ decomposing a NN into more subnetworks means that each device receives fewer parameters, as well as reduces the cost of the synchronization step, and $$ii)$$ each device trains a much smaller model, resulting in less computational costs and better energy consumption. Thus, there is good reason to expect that IST will scale much better than a “data parallel” algorithm for mobile applications. 

Results by this projects include: 

- [Distributed Learning of Deep Neural Networks using Independent Subnet Training](./IST.html).
- [GIST: Distributed Training for Large-Scale Graph Convolutional Networks](https://towardsdatascience.com/effortless-distributed-training-of-ultra-wide-gcns-6e9873f58a50).
- [ResIST: Layer-Wise Decomposition of ResNets for Distributed Training](./ResIST.html).
- [On the Convergence of Shallow Neural Network Training with Randomly Masked Neurons](./theory_IST.html)
- [LOFT: Finding Lottery Tickets through Filter-wise Training]()
- [Efficient and Light-Weight Federated Learning via Asynchronous Distributed Dropout]()

Some works that came **after this line of work**:

- [PruneFL](https://arxiv.org/pdf/1909.12326.pdf)
- [Helios](https://arxiv.org/pdf/1912.01684.pdf)
- [HeteroFL](https://arxiv.org/pdf/2010.01264.pdf)
- [FjORD - Samsung](https://arxiv.org/pdf/2102.13451.pdf)
- [PVT - Google](https://arxiv.org/pdf/2110.05607.pdf)
- [Masked NNs](https://arxiv.org/pdf/2106.08895.pdf)
- [Further theory on masked NNs](https://openreview.net/pdf?id=p3EhUXVMeyn)
- [Federated dropout - LG](https://arxiv.org/pdf/2109.15258.pdf)
- [Federated dropout - Google](https://arxiv.org/pdf/2011.04050.pdf)
- [Federated pruning - Google](https://arxiv.org/pdf/2209.06359.pdf)
- [FedSelect - Google](https://arxiv.org/pdf/2208.09432.pdf)
- [FedRolex - Google (among others)](https://arxiv.org/pdf/2212.01548.pdf)

### Publications

> Binhang Yuan, Cameron R. Wolfe, Chen Dun, Yuxin Tang, Anastasios Kyrillidis, Christopher M. Jermaine, [**``Distributed Learning of Deep Neural Networks using Independent Subnet Training''**]([https://arxiv.org/pdf/1910.02120](https://www.vldb.org/pvldb/vol15/p1581-wolfe.pdf)), Proceedings of the VLDB Endowment, Volume 15, Issue 8, April 2022, pp 1581–1590, https://doi.org/10.14778/3529337.3529343.
>
> Cameron R. Wolfe, Jingkang Yang, Arindam Chowdhury, Chen Dun, Artun Bayer, Santiago Segarra, Anastasios Kyrillidis, [**''GIST: Distributed Training for Large-Scale Graph Convolutional Networks''**](https://arxiv.org/pdf/2102.10424), Arxiv Preprint, submitted Springer special issue on Data Science and Graph Applications.
>
> Chen Dun, Cameron R. Wolfe, Christopher M. Jermaine, Anastasios Kyrillidis, [**''ResIST: Layer-wise decomposition of ResNets for distributed training''**](https://proceedings.mlr.press/v180/dun22a/dun22a.pdf), Proceedings of the Thirty-Eighth Conference on Uncertainty in Artificial Intelligence, PMLR 180:610-620, 2022.
> 
> Fangshuo (Jasper) Liao, Anastasios Kyrillidis, [**''On the convergence of shallow neural network training with randomly masked neurons''**](https://openreview.net/pdf?id=e7mYYMSyZH), Transactions on Machine Learning Research, 2022.
> 
> Qihan Wang*, Chen Dun*, Fangshuo (Jasper) Liao*, Chris Jermaine, Anastasios Kyrillidis, [**''LOFT: Finding Lottery Tickets through Filter-wise Training''**](https://arxiv.org/pdf/2210.16169.pdf), arXiv preprint arXiv:2210.16169, 2022.
> 
> Chen Dun, Mirian Hipolito, Chris Jermaine, Dimitrios Dimitriadis, Anastasios Kyrillidis, [**''Efficient and Light-Weight Federated Learning via Asynchronous Distributed Dropout''**](https://arxiv.org/pdf/2210.16105.pdf), arXiv preprint arXiv:2210.16105, 2022.
