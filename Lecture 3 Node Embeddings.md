# Lecture 3: Node Embeddings

### Review

**Graph Representation Learning Goal:** Efficient task-independent feature learning for machine learning with graphs.

<img src="https://raw.githubusercontent.com/zjwu0522/CS224W/main/images/image-20210923204625171.png" alt="image-20210923204625171" style="zoom: 67%;" />

**Why Embedding?**

- Similarity of embeddings between nodes indicates their similarity in the network. For example:
  - Both nodes are close to each other (connected by an edge)
- Encode network information
- Potentially used for many downstream predictions

## 3.1 Encoder and Decoder

### Setup

Assume we have a graph G:

- V is the vertex set;
- **A** is the adjacency matrix.

<img src="C:\Users\research\AppData\Roaming\Typora\typora-user-images\image-20210923210339146.png" alt="image-20210923210339146" style="zoom:67%;" />

### Embedding Nodes

**Goal**: encode nodes so that similarity in the embedding space (dot product for example) approximates similarity in the graph.

<img src="https://raw.githubusercontent.com/zjwu0522/CS224W/main/images/image-20210923210701360.png" alt="image-20210923210701360" style="zoom:80%;" />

### Learning Node Embeddings

- **Encoder** maps from nodes to embeddings;
- **Define a node similarity function** (a measure of similarity in the original network);
- **Decoder** maps from embeddings to the similarity score;
- Optimize the parameters of the encoder

### "Shallow" Encoding

Simplest encoding approach: **Encoder is just an embedding-lookup**.

<img src="https://raw.githubusercontent.com/zjwu0522/CS224W/main/images/image-20210923211525216.png" alt="image-20210923211525216" style="zoom: 80%;" />

<img src="C:\Users\research\AppData\Roaming\Typora\typora-user-images\image-20210923211602198.png" alt="image-20210923211602198" style="zoom:80%;" />

In this approach, **Each node is assigned a unique embedding vector.**

**Pro:** If we obtain **Z**, we can easily get node embedding

**Con:** Too many parameter to be learned, difficult to scale up on large graphs

### Summary

- **Encoder + Decoder Framework**
  - Shallow encoder: embedding lookup
  - Parameter to optimize: **Z** which contains node embedding for all nodes
  - **Decoder:** based on node similarity
  - **Objective:** maximize score for node pairs (u,v) that are similar

- **Note on Node Embeddings**
  - This is **unsupervised** way of learning no embeddings
    - no node labels
    - no node features
    - directly estimate the embedding of a node so some aspect of the network structure is preserved
  - These embeddings are **task independent**

## 3.2 Random Walk Approaches for Node Embeddings

### Notations

<img src="C:\Users\research\AppData\Roaming\Typora\typora-user-images\image-20210923224214787.png" alt="image-20210923224214787" style="zoom:80%;" />

### Random Walk

Given a *Graph* and a *starting point*, we **select a neighbor** of it at **random**, and move to this neighbor; then we select a neighbor of this point at random, and move to it, etc. The (random) sequence of points visited this way is a random walk on the graph.

#### Random Walk Embeddings

![image-20210923224441455](https://raw.githubusercontent.com/zjwu0522/CS224W/main/images/image-20210923224441455.png)

- Estimate probability of visiting node v on a random walk starting from node u using some random walk Strategy **R**;
- Optimize embeddings to encode these random walk statistics.

#### Why Random Walks?

- Expressivity: Flexible stochastic definition of node similarity that incorporates both local and higher-order neighborhood information 
  - Idea: if random walk starting from node u visits v with high probability, u and v are similar (high-order multi-hop information)

- **Efficiency**: Do not need to consider all node pairs when training; only need to consider pairs that co-occur on random walks

#### Unsupervised Feature Learning

- **Intuition**: Find embedding of nodes in d-dimensional space that preserves similarity
- **Idea**: Learn node embedding such that nearby nodes are close together in the network

- Given a node u, how do we define nearby nodes?

<img src="C:\Users\research\AppData\Roaming\Typora\typora-user-images\image-20210923225018445.png" alt="image-20210923225018445" style="zoom:80%;" />

#### Optimization

1. Run **short fixed-length random walks** starting from each node u in the graph using some random walk strategy **R**;
2. For each node u collect N_R(u), the multiset of nodes visited on random walks starting from u;
3. Optimize embeddings according to: Given a node u, predict its neighbors N_R(u).

<img src="C:\Users\research\AppData\Roaming\Typora\typora-user-images\image-20210923225745879.png" alt="image-20210923225745879" style="zoom:67%;" />

<img src="C:\Users\research\AppData\Roaming\Typora\typora-user-images\image-20210923225809896.png" alt="image-20210923225809896" style="zoom:67%;" />

<img src="C:\Users\research\AppData\Roaming\Typora\typora-user-images\image-20210923225949618.png" alt="image-20210923225949618" style="zoom:80%;" />

However, to compute this loss function, the time complexity is too high so we need optimize it, We found that the problem lies in the denominator used for SoftMax normalization:

<img src="C:\Users\research\AppData\Roaming\Typora\typora-user-images\image-20210923230248095.png" alt="image-20210923230248095" style="zoom:80%;" />

This random distribution is not uniform random, but random in a biased way: the probability is proportional to its degree.

Considerations for the number of negative samples k:

- Higher k gives more robust estimates
- However, higher k corresponds to higher bias on negative events
- in practice, k ranges from 5 to 20

After we obtained the objective function, how do we optimiaze it?

- **Gradient Descent** 

- **Stochastic Gradient Descent**

So now we have learnt hoe to optimize embeddings given a random walk strategy **R**, then what strategies should we use to run these random walks?

- Simplest idea: **Just run fixed-length, unbiased random walks starting from each node ([DeepWalk](https://arxiv.org/abs/1403.6652)).**

  <img src="C:\Users\research\AppData\Roaming\Typora\typora-user-images\image-20210924152834045.png" alt="image-20210924152834045" style="zoom:67%;" />

  - The issue is that such notion of similarity is too constrained
  - We can use **Node2vec** to generalize this

### Node2vec

**Goal:** Embed nodes with similar network neighborhoods close in the feature space

**Why:**

- Flexible notion of network neighborhood N_R(u) of node u leads to rich node embeddings
- Use biased walks that can trade off between **local** and **global** views of the nework

<img src="C:\Users\research\AppData\Roaming\Typora\typora-user-images\image-20210924164142844.png" alt="image-20210924164142844" style="zoom:67%;" />

#### Biased Fixed-length Random Walk

Given a node u, it can generate its neighborhood N_R(u).

Tow parameters:

- Return parameter p:
  - Return back to the previous node
- In-out parameter q:
  - Moving outwards (DFS) vs. inwards (BFS)
  - Intuitively, q is the ration of BFS vs. DFS

**Example**

<img src="C:\Users\research\AppData\Roaming\Typora\typora-user-images\image-20210924164657102.png" alt="image-20210924164657102" style="zoom:80%;" />

<img src="C:\Users\research\AppData\Roaming\Typora\typora-user-images\image-20210924164718845.png" alt="image-20210924164718845" style="zoom:80%;" />

#### Node2vec Algorithm

1. Compute random walk probabilities
2. Simulate r random walks of length l starting from each node u
3. Optimize the node2vec objective using Stochastic Gradient Descent

**Pros:** Linear-time complexity since the node neighbors are fixed; all 3 steps are individually parallelizable

### Summary

**Core idea:** Embed nodes so that distances in embedding space reflect node similarities in the original network.

**Different notions of node similarity:**

- Naïve: similar if 2 nodes are connected
- Neighborhood overlap
- Random walk approaches

**In general:** must choose definition of node similarity that matches our application

## 3.3 Embedding Entire Graphs

**Goal:** Want to embed a subgraph or an entire graph **G**. Graph embedding: Z_G.

<img src="https://raw.githubusercontent.com/zjwu0522/CS224W/main/images/image-20210925155230447.png" alt="image-20210925155230447" style="zoom: 67%;" />

- **Approach 1**
  - Run a standard graph embedding technique on the (sub)graph G;
  - just sum (or average) the node embeddings in the (sub)graph G;
  - used by  [Duvenaud et al., 2016](https://arxiv.org/abs/1509.09292) to classify molecules based on their graph structure;
  - simple but successful.

- **Approach 2**
  - Introduce a **virtual node** to represent the (sub)graph and run a standard graph embedding technique;
  - proposed by [Li et al, 2016](https://arxiv.org/abs/1511.05493) as a general technique for subgraph embedding.

<img src="C:\Users\research\AppData\Roaming\Typora\typora-user-images\image-20210925161650141.png" alt="image-20210925161650141" style="zoom:67%;" />

- **Approach 3 Anonymous Walk Embeddings**

  - [Anonymous Walk Embeddings, ICML 2018](https://arxiv.org/pdf/1805.11921.pdf)

  - States in **anonymous walks **correspond to the index of the **first time** we visited the node in a random walk;

    <img src="https://raw.githubusercontent.com/zjwu0522/CS224W/main/images/image-20210925161834316.png" alt="image-20210925161834316" style="zoom:67%;" />

  - agnostic to the identity of the nodes visited (hence anonymous);

  - number of walks grows exponentially regarding length;

  - simple use:

    - Simulate anonymous walks of l steps and record their counts;
    - represent the graph as a probability distribution over these walks;
    - to some extent similar to **bag of anonymous walks**.
    - For example:

<img src="C:\Users\research\AppData\Roaming\Typora\typora-user-images\image-20210925162309774.png" alt="image-20210925162309774" style="zoom:67%;" />

<img src="C:\Users\research\AppData\Roaming\Typora\typora-user-images\image-20210925162458089.png" alt="image-20210925162458089" style="zoom:67%;" />

### Learn Walk Embeddings

**Key Idea:** Rather than simply represent each walk by the fraction of times it occurs, we **learn embedding z_i of anonymous walk w_i**.

That is, learn a graph embedding Z_G together with all the anonymous walk embeddings z_i, where Z = {z_i: i = 1 ... n }, n is the number of sampled anonymous walks.

<img src="C:\Users\research\AppData\Roaming\Typora\typora-user-images\image-20210925164002807.png" alt="image-20210925164002807" style="zoom:80%;" />

<img src="C:\Users\research\AppData\Roaming\Typora\typora-user-images\image-20210925164022353.png" alt="image-20210925164022353" style="zoom:80%;" />

- We obtain the graph embedding Z_G after optimization
- Use Z_G to make predictions such as graph classification
  - Option 1: inner product kernel
  - Option 2: use a neural network that takes Z_G as input to classify
- Overall Arch:

<img src="C:\Users\research\AppData\Roaming\Typora\typora-user-images\image-20210925164159820.png" alt="image-20210925164159820" style="zoom:67%;" />

### How to Use Embeddings

<img src="C:\Users\research\AppData\Roaming\Typora\typora-user-images\image-20210925164516629.png" alt="image-20210925164516629" style="zoom:80%;" />

### Summary

- We discussed 3 ideas to graph embeddings
- For approach 3, we have two ideas:
  - Sample the anonymous walks and represent the graph as fraction of times each anon walk occurs
  - Embed anonymous walks, concatenate their embeddings to get a graph embedding

## Conclusion

We discussed **graph representation learning**, a way to learn **node and graph embeddings** for downstream tasks, **without feature engineering**.

- Encoder-decoder framework:
  - Encoder: embedding lookup
  - Decoder: predict score based on embedding to match node similarity
- Node similarity measure: (biased) random walk
  - Examples: DeepWalk, Node2Vec
- Extension to Graph embedding:
  - Node embedding aggregation and Anonymous Walk Embeddings























