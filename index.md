---
layout: default
---

Neural networks (NN) are ubiquitous, and have led to the recent success of machine learning (ML). For many applications, model training and inference need to be carried out in a timely fashion on a computation-/communication-light, and energy-limited platform. Optimizing the NN training in terms of computation/com- munication/energy efficiency, while retaining competitive accuracy, has become a fundamental challenge.

We propose a new class of neural network (NN) training algorithms, called independent subetwork training (IST). IST decomposes the NN into a set of independent subnetworks. Each of those subnetworks is trained at a different device, for one or more backpropagation steps, before a synchronization step. Updated subnetworks are sent from edge-devices to the parameter server for reassembly into the original NN, before the next round of decomposition and local training. Because the subnetworks share no parameters, synchronization requires no aggregation; it is just an exchange of parameters. Moreover, each of the subnetworks is a fully-operational classifier by itself; no synchronization pipelines between subnetworks are required. Key attributes of our proposal is that $$i)$$ decomposing a NN into more subnetworks means that each device receives fewer parameters, as well as reduces the cost of the synchronization step, and $$ii)$$ each device trains a much smaller model, resulting in less computational costs and better energy consumption. Thus, there is good reason to expect that IST will scale much better than a “data parallel” algorithm for mobile applications. 

Results by this projects include: 

- [Distributed Learning of Deep Neural Networks using Independent Subnet Training](./IST.html).
- [GIST: Distributed Training for Large-Scale Graph Convolutional Networks](./GIST.html).
- [ResIST: Layer-Wise Decomposition of ResNets for Distributed Training](./ResIST.html).

### Publications

> Binhang Yuan, Cameron R. Wolfe, Chen Dun, Yuxin Tang, Anastasios Kyrillidis, Christopher M. Jermaine, [**``Distributed Learning of Deep Neural Networks using Independent Subnet Training''**](https://arxiv.org/pdf/1910.02120), Arxiv Preprint, arXiv preprint arXiv:1910.02120 (2019).
>
> Cameron R. Wolfe, Jingkang Yang, Arindam Chowdhury, Chen Dun, Artun Bayer, Santiago Segarra, Anastasios Kyrillidis, [**''GIST: Distributed Training for Large-Scale Graph Convolutional Networks''**](https://arxiv.org/pdf/2102.10424), Arxiv Preprint, arXiv preprint arXiv:2102.10424 (2021).
>
> Chen Dun, Cameron R. Wolfe, Christopher M. Jermaine, Anastasios Kyrillidis, [**''ResIST: Layer-Wise Decomposition of ResNets for Distributed Training''**](), Arxiv Preprint, arXiv preprint arXiv:xxxx.xxxx (2021).
