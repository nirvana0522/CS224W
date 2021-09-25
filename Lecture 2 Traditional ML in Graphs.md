# Lecture 2: Traditional ML in Graphs

## 2.1 Traditional Feature-based Methods: Node-level

### Review

Traditional ML pipeline: design and obtain features for all training data regrading nodes, links and graphs, then train ML model to apply it to make a prediction.

Graph data itself has features, but we are also interested in features that describe its local structure features, that is, the feature describes the topology of the network and structure features can help to make more accurate predictions.

So generally there are two kinds of features:

- Structural features;
- Attributes and properties of nodes or relations.

![image-20210921154158996](https://raw.githubusercontent.com/nirvana0522/CS224W/main/images/image-20210921154158996.png)

### Design Choice

- Features: d-dimensional vectors;
- Objects: Nodes, edges, sets of nodes, entire graphs;
- Objective functions: what tasks are we aiming to solve?

### Node-Level Features:

##### Goal: Characterize the structure and position of a node in the network:

- Node degree
- Node centrality
- Clustering coefficient 
- Graphlets

<img src="https://raw.githubusercontent.com/nirvana0522/CS224W/main/images/image-20210921155333843.png" alt="image-20210921155333843" style="zoom:67%;" />

##### Node Degree

The degree $$k_v$$ of node $$v$$ is the number of edges (neighboring nodes) the nodes has.

##### Node Centrality

- Why: node degree treat all neighboring nodes equally so it cannot capture their importance;
- Node Centrality $$c_v$$ takes the node importance in a given graph into account;
- Different ways to model importance

  1. Eigenvector

  <img src="https://raw.githubusercontent.com/nirvana0522/CS224W/main/images/image-20210921160740783.png" alt="image-20210921160740783" style="zoom:80%;" />

  ![image-20210921160802812](https://raw.githubusercontent.com/nirvana0522/CS224W/main/images/image-20210921160802812.png)

  2. Betweenness

  <img src="https://raw.githubusercontent.com/nirvana0522/CS224W/main/images/image-20210921160822839.png" alt="image-20210921160822839" style="zoom:80%;" />

  3. Closeness

  ![image-20210921160835135](https://raw.githubusercontent.com/nirvana0522/CS224W/main/images/image-20210921160835135.png)

##### Clustering Coefficient

It measures how connected $$v's$$ neighboring nodes are:

![image-20210921161056037](https://raw.githubusercontent.com/nirvana0522/CS224W/main/images/image-20210921161056037.png)

The first example:  $$e_v = \frac{6}{6}$$
The second example: $$e_v = \frac{3}{6}$$
The third example: $$e_v = \frac{0}{6}$$

##### Graphlets

**Observation:** Clustering coefficient counts the #(triangles) in the ego-network

<img src="https://raw.githubusercontent.com/nirvana0522/CS224W/main/images/image-20210921161427911.png" alt="image-20210921161427911" style="zoom: 67%;" />

**Ego-network** of a given node is simply a network that is induced by the node itself and its neighbors. So it's  basically degree 1 neighborhood network around a a given node.

We can generalize the above by counting #(pre-specified subgraphs, i.e., **graphlets**)

There are many such triangles in social networks, because it is conceivable that your friends may know each other through your introduction, thus constructing such a triangle/triple.

This triangle can be extended to some predefined subgraph pre-specified subgraph, such as the graphlet shown below:

<img src="https://raw.githubusercontent.com/nirvana0522/CS224W/main/images/image-20210921162630613.png" alt="image-20210921162630613" style="zoom:67%;" />

- **Graphlet Degree Vector (GDV):** Graphlet-based features for nodes, it counts **graphlets** that a nods touches.;
- **Degree** counts **edges** that a node touches;
- **Clustering coefficient** counts **triangles** that a node touches

<img src="https://raw.githubusercontent.com/nirvana0522/CS224W/main/images/image-20210921162941168.png" alt="image-20210921162941168" style="zoom:80%;" />



### Node-Level Summary

- Importance-based features:

  - Node degree
  - Different node centrality measures

  It can be used to predict celebrity users in a social network.

- Structure-based features:

  -  Node degree
  - Clustering coefficient
  - Graphlet count vector 

  It can be used to predict protein functionality in a PPI network.



## 2.2 Traditional Feature-based Methods: Link-level

Link prediction tasks can be formulated as predicting new links given existing links. When testing, all node pairs (no existing links) are ranked, and top K node pairs are predicted. Key challenge is to design a pair of nodes.

<img src="https://raw.githubusercontent.com/nirvana0522/CS224W/main/images/image-20210922210719124.png" alt="image-20210922210719124" style="zoom:80%;" />

There are two different tasks:

- Links missing at random
- Links over time

<img src="https://raw.githubusercontent.com/nirvana0522/CS224W/main/images/image-20210922210949454.png" alt="image-20210922210949454" style="zoom:80%;" />

### Link Prediction via Proximity

<img src="https://raw.githubusercontent.com/nirvana0522/CS224W/main/images/image-20210922211158092.png" alt="image-20210922211158092" style="zoom:80%;" />

### Link-Level Features

1. **Distance-based Feature**

   it measures the shortest-path distance between tow nods.

<img src="https://raw.githubusercontent.com/nirvana0522/CS224W/main/images/image-20210922214332270.png" alt="image-20210922214332270" style="zoom:80%;" />

​	Cons: it does not capture the degree of neighborhood overlap.

2. **Local Neighborhood Overlap**

   It captures neighborhood bodes shared between two nodes $$v1$$ and $$v2$$.

<img src="https://raw.githubusercontent.com/nirvana0522/CS224W/main/images/image-20210922214540814.png" alt="image-20210922214540814" style="zoom:67%;" />

​	The problem with common neighbors is that point pairs with higher degrees will have higher results, and 	Jaccard’s coefficient is the normalized result.

​	The Adamic-Adar index performs well in practice. The reason for performing well on social networks: A 	group of low-degree mutual friends scores higher than a group of celebrity mutual friends.

3. **Global Neighborhood Overlap**

   Limitation of local neighborhood features:

   - Metric is always zero if the two nodes do not have any neighbors in common;
   - the two nodes may still potentially be connected in the future.

   Global neighborhood overlap metrics resolve the limitation by considering the entire graph.

   **Katz index:** count the number of paths of all lengths between a given pair of nodes.

   **How to use graph adjacency matrix to compute paths between tow nodes:**

   <img src="https://raw.githubusercontent.com/nirvana0522/CS224W/main/images/image-20210922215846820.png" alt="image-20210922215846820" style="zoom:80%;" />

<img src="https://raw.githubusercontent.com/nirvana0522/CS224W/main/images/image-20210922220010578.png" alt="image-20210922220010578" style="zoom:80%;" />

**Discount factor $$\beta$$** will assign a smaller weight to long distance path exponentially.

<img src="https://raw.githubusercontent.com/nirvana0522/CS224W/main/images/image-20210922220329662.png" alt="image-20210922220329662" style="zoom:67%;" />

### Summary

- Distance-based features:

   Uses the shortest path length between two nodes but does not capture how neighborhood overlaps.

- Local neighborhood overlap:
  - Captures how many neighboring nodes are shared by two nodes.
  - Becomes zero when no neighbor nodes are shared.

- Global neighborhood overlap:
  - Uses global graph structure to score two nodes.
  - Katz index counts #paths of all lengths between two nodes.

## 2.3 Traditional Feature-based Methods: Graph-level

**Goal: We want features that characterize the structure of an entire graph**

### Kernel Methods

Idea: Design **Kernels instead of feature vectors.**

A quick introduction to Kernels:

- Kernel $$K(G,G\prime \in R) $$ measures similarity b/w data
- Kernel $$\mathbf{K} = (K(G,G\prime))_{G,G\prime} $$ must always be positive semidefinite (has positive eigenvals).
- There exists a feature representation $$\phi()$$ such that $$K(G,G\prime) = \phi(G)^{T}\phi(G\prime)$$.
- Once the kernel is defined, off-the-shelf ML model, such as kernel SVM, can be used to make predictions.

$$\phi()$$ is a representation vector, which may not need to be calculated explicitly.

### Graph Kernel: Key Idea

**Goal: Design graph feature vector $$\phi(G)$$** 

Use **Bag of Words (BoW)** for a graph. BoW simply uses the word counts as features for documents.

Naïve extension to a graph: **Regards nodes as words.**

For following two graphs which have 4 red nodes, we get the same feature vector for these two graphs:

<img src="https://raw.githubusercontent.com/nirvana0522/CS224W/main/images/image-20210922231014885.png" alt="image-20210922231014885" style="zoom: 67%;" />

So we could consider using **Bag of node degrees**.

Both **Graphlet Kernel** and **Weisfeiler-Lehman Kernel** use **Bag of *** representation of graph, where ***** is more sophisticated than node degrees.

### Graphlet Features

- **Key idea:** Count the number of different graphlets in a graph
- **Difference from node-level features:**
  - Nodes in graphlets here do not need to be connected (allows for isolated nodes);
  - The graphlets here are not rooted.

<img src="https://raw.githubusercontent.com/nirvana0522/CS224W/main/images/image-20210922231510223.png" alt="image-20210922231510223" style="zoom:67%;" />

- **Graphlet count vector:** Given graph $$G$$, and a graphlet list $$G_{k} = (g_{1},g_{2},...,g_{n_{k}})$$, define the graphlet count vector as:

<img src="https://raw.githubusercontent.com/nirvana0522/CS224W/main/images/image-20210922231752354.png" alt="image-20210922231752354" style="zoom:67%;" />

- **Example:**

  <img src="https://raw.githubusercontent.com/nirvana0522/CS224W/main/images/image-20210922231820188.png" alt="image-20210922231820188" style="zoom:67%;" />

- For different sizes that will greatly skew the value, we normalize each feature vactor:

  <img src="https://raw.githubusercontent.com/nirvana0522/CS224W/main/images/image-20210922231930249.png" alt="image-20210922231930249" style="zoom:67%;" />

- **Limitations:** Counting graphlets is expensive.

  

### Weisfeiler-Lehman Kernel

**Goal: design an efficient graph feature descriptor $$\phi(G)$$**.

**Idea:** use neighborhood structure to iteratively enrich node vocabulary.

##### Color Refinement 

- Assign an initial color to each node $$v$$ and then iteratively refine node colors by

  <img src="https://raw.githubusercontent.com/nirvana0522/CS224W/main/images/image-20210922232252128.png" alt="image-20210922232252128" style="zoom:67%;" />

  After K steps of refinement, $$c^{K}(v)$$ summarizes the structure of K-hop neighborhood.

- **Example**

<img src="https://raw.githubusercontent.com/nirvana0522/CS224W/main/images/image-20210922232354855.png" alt="image-20210922232354855" style="zoom:67%;" />

<img src="https://raw.githubusercontent.com/nirvana0522/CS224W/main/images/image-20210922232403609.png" alt="image-20210922232403609" style="zoom:67%;" />

<img src="https://raw.githubusercontent.com/nirvana0522/CS224W/main/images/image-20210922232439837.png" alt="image-20210922232439837" style="zoom:67%;" />

<img src="https://raw.githubusercontent.com/nirvana0522/CS224W/main/images/image-20210922232507061.png" alt="image-20210922232507061" style="zoom:67%;" />

<img src="https://raw.githubusercontent.com/nirvana0522/CS224W/main/images/image-20210922232517652.png" alt="image-20210922232517652" style="zoom:67%;" />

- **Pros**: Computationally efficient

### Summary

- Graphlet Kernel
  - Graph is represented as **Bag of graphlets**
  - **Computationally expensive**
- Weisfeiler-Lehman Kernel
  - Apply K-step color refinement algorithm to enrich node colors
    - Different colors capture different K-hop neighborhood structures
  - Graph is represented as **Bag of colors**
  - **Computationally efficient**

## Conclusion

- Traditional ML Pipeline
  - Hand-crafted feature $$+$$ ML model
- Hand-crafted features for graph data
  - **Node-level**
    - Node degree, centrality, clustering coefficient, graphlets
  - **Link-level**
    - Distance-based feature
    - local/global neighborhood overlap
  - **Graph-level**
    - Graphlet kernel, Weisfeiler-Lehman Kernel

































































































 

