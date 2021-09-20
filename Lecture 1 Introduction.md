# Lecture 1: Introduction

## 1.1 Why Graphs

### Perquisite

- Machine Learning
- Algorithms and graph theory
- Probability and statistics 

### Graph Machine Learning Tools

1. [Pytorch Geometric (PYG)](https://github.com/rusty1s/pytorch_geometric)
2. [DeepSNAP](https://github.com/snap-stanford/deepsnap): Library that assists deep learning on graphs. 
3. [GraphGym](https://github.com/snap-stanford/GraphGym): Platform for designing graph neural Networks

### Many Types of Graph

![image-20210920211633803](C:\Users\research\AppData\Roaming\Typora\typora-user-images\image-20210920211633803.png)



### Why is it Hard?

![image-20210920211731616](C:\Users\research\AppData\Roaming\Typora\typora-user-images\image-20210920211731616.png)

### Deep Learning in Graphs

![image-20210920211842017](C:\Users\research\AppData\Roaming\Typora\typora-user-images\image-20210920211842017.png)

### Supervised Machine Learning Lifecycle

In traditional ML pipeline, we need to apply feature engineering on the raw data, such as manually extracting features. Now we can use representation learning methods to automatically learn this features, and apply them directly to downstream prediction tasks.

![image-20210920213252227](C:\Users\research\AppData\Roaming\Typora\typora-user-images\image-20210920213252227.png)

### Representation Learning

Representation Learning aims to map nodes to a d- dimensional **embeddings** such that **similar nodes in the network are embedded close together.** 

![image-20210920213452578](C:\Users\research\AppData\Roaming\Typora\typora-user-images\image-20210920213452578.png)



## 1.2 Applications of Graph ML

Generally, there are 4 main different types of Graph ML tasks:

![image-20210920213706441](C:\Users\research\AppData\Roaming\Typora\typora-user-images\image-20210920213706441.png)

1. Node level
2. Edge level
3. Community/subgraph level
4. Graph level, including graph-level prediction and graph generation

### Example of Node-level ML Tasks

Computationally predict a protein's 3D structure based solely on its amino acid sequence: **AlphaFold**

<img src="C:\Users\research\AppData\Roaming\Typora\typora-user-images\image-20210920214637160.png" alt="image-20210920214637160"  />

### Example of Edge-level ML Tasks

Recommender Systems: **PinSage**

The task is to recommend items users might like

![image-20210920215045797](C:\Users\research\AppData\Roaming\Typora\typora-user-images\image-20210920215045797.png)

### Example of Edge-level ML Tasks

Google Map predicts the length and time-consuming of a certain distance: model the road section into a graph, and establish a prediction model on each subgraph.

![image-20210920215836665](C:\Users\research\AppData\Roaming\Typora\typora-user-images\image-20210920215836665.png)

### Example of Graph-level ML Tasks

![image-20210920220100770](C:\Users\research\AppData\Roaming\Typora\typora-user-images\image-20210920220100770.png)



## 1.3 Choice of Graph Representation

### Component of a Network

- **Objects:** nodes, vertices	  **N**
- **Interactions:** links, edges    **E**
- **System:** network, graph     **G(N,E)**

![image-20210920221004501](C:\Users\research\AppData\Roaming\Typora\typora-user-images\image-20210920221004501.png)

Graphs is a common language for solving relational problems, a unified mathematical representation in various situations. By abstracting the problem into a graph, all problems can be solved with the same machine learning algorithm

![image-20210920221318815](C:\Users\research\AppData\Roaming\Typora\typora-user-images\image-20210920221318815.png)



### Choosing a Proper Representation 

- If you connect individuals that work with each other, you will explore a **professional network**;
- If you connect those that have a sexual relationship, you will be exploring **sexual networks**;
- If you connect scientific papers that cite each other, you will be studying the **citation network**.

### How do you define a graph

- How to build a graph:

  What are nodes?

  What are edges?

- Choice the proper representation of a given domain determines our ability to use networks
  1.  In some cases there is a unique, unambiguous representation;
  2.  In other cases, the representation is by no means unique;
  3.  The way you assign links will determine the nature of the question you can study.

### Directed vs. Undirected Graphs

Some examples of designing choices we care faced with when creating graphs

![image-20210920222031260](C:\Users\research\AppData\Roaming\Typora\typora-user-images\image-20210920222031260.png)

**Node Degree, $k_i$:** the number of edges adjacent to node **i** 

**Avg. degree:** 
$$
\overline{k} = <k> = \frac{1}{N}\sum_{i=1}^{N}k_i = \frac{2E}{N}
$$
![image-20210920224019756](C:\Users\research\AppData\Roaming\Typora\typora-user-images\image-20210920224019756.png)

In directed networks we define an **in-degree** and **out-degree**. The total degree of a node is the sum of in- and out-degrees.

![image-20210920224214775](C:\Users\research\AppData\Roaming\Typora\typora-user-images\image-20210920224214775.png)

![image-20210920224222114](C:\Users\research\AppData\Roaming\Typora\typora-user-images\image-20210920224222114.png)

### Bipartite Graph

![image-20210920224601999](C:\Users\research\AppData\Roaming\Typora\typora-user-images\image-20210920224601999.png)

### Adjacency Matrix

![image-20210920224950594](C:\Users\research\AppData\Roaming\Typora\typora-user-images\image-20210920224950594.png)



![image-20210920225001242](C:\Users\research\AppData\Roaming\Typora\typora-user-images\image-20210920225001242.png)



Normally, Adjacency matrix are Sparse.

### Representing Graphs

- Edge List: This method is often used in deep learning frameworks because the graph can be directly represented as a two-dimensional matrix. The problem with this representation method is that it is difficult to operate and analyze the graph, even if it is just to calculate the degree of the node in the graph, it will be difficult.

<img src="C:\Users\research\AppData\Roaming\Typora\typora-user-images\image-20210920225616465.png" alt="image-20210920225616465" style="zoom: 67%;" />

- **Adjacency list:** A much better representation for a graph analysis and manipulation is the notion of adjacency list. 

<img src="C:\Users\research\AppData\Roaming\Typora\typora-user-images\image-20210920225717504.png" alt="image-20210920225717504" style="zoom:67%;" />



### Node and Edge Attributes

**Possible options:** Not only topological nodes and edges, but also their attributes.

-  Weight (e.g. frequency of communication) 
-  Ranking (best friend, second best friend…) § Type (friend, relative, co-worker) 
-  Sign: Friend vs. Foe, Trust vs. Distrust 
-  Properties depending on the structure of the rest of the graph: Number of common friends

### Many Types of Graphs

![image-20210920230105130](C:\Users\research\AppData\Roaming\Typora\typora-user-images\image-20210920230105130.png)

![image-20210920230232666](C:\Users\research\AppData\Roaming\Typora\typora-user-images\image-20210920230232666.png)



### Connectivity of Undirected Graphs

![image-20210920230415789](C:\Users\research\AppData\Roaming\Typora\typora-user-images\image-20210920230415789.png)

![image-20210920230508514](C:\Users\research\AppData\Roaming\Typora\typora-user-images\image-20210920230508514.png)

### Connectivity of Directed Graphs

![image-20210920230609221](C:\Users\research\AppData\Roaming\Typora\typora-user-images\image-20210920230609221.png)

<img src="C:\Users\research\AppData\Roaming\Typora\typora-user-images\image-20210920230637911.png" alt="image-20210920230637911" style="zoom:80%;" />

## Summary

- Machine Learning with Graphs

  Applications and use cases

-  Different Types of Tasks:
  - Node level
  - Edge level
  - Graph level
- Choice of a graph representation
  -  Directed, undirected, bipartite, weighted, adjacency matrix







