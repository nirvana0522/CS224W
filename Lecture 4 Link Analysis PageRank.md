# Lecture 4: Link Analysis: PageRank

### Graph as Matrix

- This lecture introduces graph analysis and learning from a matrix perspective
- The importance of nodes can be defined by means of random walk (PageRank), and node embedding can be obtained by matrix factorization

## PageRank: AKA the Google Algorithm

- Web as a graph:
  - Nodes = web pages
  - Edges = hyperlinks
- Side issue (not discussed in this lecture):
  - Dynamic pages created on the fly
  - 'dark matter' - inaccessible database generated pages

- Web as a **directed graph**:

<img src="https://raw.githubusercontent.com/zjwu0522/CS224W/main/images/image-20211009163052406.png" alt="image-20211009163052406" style="zoom:67%;" />

### Link Analysis Algorithms

- PageRank
- Personalized PageRank (PPR)
- Random Walk with Restarts

#### Idea: Links as votes

- In-coming links
- Out-coming links
- Links from important pages count more, use recursive to obtain importance

#### Flow Model

- Each link’s vote is proportional to the importance of its source page
- If page i with importance r_i has di out-links, each link gets r_i / d_i votes
- Page j’s own importance r_j is the sum of the votes on its in-links

<img src="C:\Users\research\AppData\Roaming\Typora\typora-user-images\image-20211009163557513.png" alt="image-20211009163557513" style="zoom:67%;" />

- A page is important if it is pointed to by other important pages

![image-20211009163635136](https://raw.githubusercontent.com/zjwu0522/CS224W/main/images/image-20211009163635136.png)

<img src="https://raw.githubusercontent.com/zjwu0522/CS224W/main/images/image-20211009163647935.png" alt="image-20211009163647935" style="zoom:67%;" />

### PageRank: Matrix Formulation

**Stochastic adjacency matrix M**

- Let page j have d_j out-links

- if j -> i, then M_ij = 1 / d_j
- Columns sum to 1
- **Rank vector r:** An entry per page, the importance score of page i, sum(r_i) = 1
- **The flow equations can be written as r = M ⋅ r**

<img src="https://raw.githubusercontent.com/zjwu0522/CS224W/main/images/image-20211009164057051.png" alt="image-20211009164057051" style="zoom:67%;" />

- Example

<img src="https://raw.githubusercontent.com/zjwu0522/CS224W/main/images/image-20211009164123874.png" alt="image-20211009164123874" style="zoom:80%;" />

### Connection to Random Walk

Imagine a random web surfer:

1. At any time **t**, surfer is on some page **i**;
2. At time **t + 1**. the surfer follows an out-link from **i** uniformly at random;
3. Ends up on some page j linked from **i**;
4. process repeats indefinitely.

In this case, **p(t)** vector whose ith coordinate is the probability that the surfer at page i at time t. In other word, **p(t)** is a probability distribution over pages.

##### The stationary distribution

Use **p(t + 1) = M ⋅ p(t)**, suppose the random walk reaches a state **p(t + 1) = M ⋅ p(t) = p(t)**, then **p(t)** is **stationary distribution** of a random walk.

Back to our original rank vector **r**, it satisfies **r = M ⋅ r**, so **r** is a stationary distribution for the random walk.

<img src="https://raw.githubusercontent.com/zjwu0522/CS224W/main/images/image-20211010162722126.png" alt="image-20211010162722126" style="zoom:67%;" />

### Eigenvector Formulation

Recall from lecture 2, 

<img src="https://raw.githubusercontent.com/zjwu0522/CS224W/main/images/image-20211010163514191.png" alt="image-20211010163514191" style="zoom: 80%;" />

This is the definition of eigenvector centrality for undirected graphs;

PageRank is defined for directed graphs.

<img src="https://raw.githubusercontent.com/zjwu0522/CS224W/main/images/image-20211010163659002.png" alt="image-20211010163659002" style="zoom:80%;" />

Rank vector **r** is the eigenvector of stochastic matrix **M**, and eigenvalue lambda is 1.

Starting from a random vector **u**, after long-term computation, the limit **M(M(… M(Mu))))** will converge to **r**, which is the principal eigenvector of **M**.

Now we can find a way to solve **r**, to get the result of PageRank.

### Summary

- PageRank measures importance of nodes in a graph using the link structure of the web
- It models a random web surfer using the stochastic adjacency matrix **M**
- PageRank solves **r = Mr** where **r** can be viewed as both the **principle eigenvector** of **M** and as the **stationary distribution of a random walk** over the graph

## How to Solve PageRank

Given a graph with n nodes, we use an iterative procedure:

- Assign each node an initial page rank
- Repeat until convergence: sum(abs(r[t+1] - r[t]) < epsilon 
  - Calculate the page rank of each node

<img src="https://raw.githubusercontent.com/zjwu0522/CS224W/main/images/image-20211010183157080.png" alt="image-20211010183157080" style="zoom:67%;" />

**Power Iteration Method**

Given a web graph with N nodes, where the nodes are pages and edges are hyperlinks

<img src="https://raw.githubusercontent.com/zjwu0522/CS224W/main/images/image-20211010191354689.png" alt="image-20211010191354689" style="zoom:67%;" />

Where |x|_1 = sum(x_i) is the **L_1** norm.

**Example**

<img src="https://raw.githubusercontent.com/zjwu0522/CS224W/main/images/image-20211010191511754.png" alt="image-20211010191511754" style="zoom:80%;" />

### Problems and Solutions Towards PageRank

1. Does this converge?
2. Does it converge to what we want?
3. Are results reasonable?

#### **Problems**

- **Dead ends:** nodes that have no out-links, will cause importance *leak out*
- **Spider traps:** all out-links are within the group, eventually spider traps absorb all importance

#### **Solutions**

- **Dead end**

  <img src="https://raw.githubusercontent.com/zjwu0522/CS224W/main/images/image-20211010191934039.png" alt="image-20211010191934039" style="zoom:67%;" />

  - **Teleports:** Follow random teleport links with total probability **1.0** from dead ends

<img src="https://raw.githubusercontent.com/zjwu0522/CS224W/main/images/image-20211010192049613.png" alt="image-20211010192049613" style="zoom:67%;" />

- **Spider trap**
  - At each time step, the random surfer has two options:
    1. with probability *beta*, follow a link at random
    2. with probability *1- beta*, jump to a random page
    3. common values for *beta* are in the range 0.8 to 0.9

<img src="https://raw.githubusercontent.com/zjwu0522/CS224W/main/images/image-20211010192628581.png" alt="image-20211010192628581" style="zoom:67%;" />

#### Why teleports solve the problem

- Spider-traps are not a problem, but with traps PageRank scores are not what we want
  - Solution: Never get stuck in a spider trap by teleporting out of it in a finite number of steps

- Dead-ends are a problem
  - The matrix is not column stochastic so our initial assumptions are not met
  - Solution: Make matrix column stochastic by always teleporting when there is nowhere else to go

### The Google Matrix

<img src="https://raw.githubusercontent.com/zjwu0522/CS224W/main/images/image-20211010193243768.png" alt="image-20211010193243768" style="zoom:80%;" />

<img src="https://raw.githubusercontent.com/zjwu0522/CS224W/main/images/image-20211010193300320.png" alt="image-20211010193300320" style="zoom:80%;" />

### Summary

- **PageRank** solves for **r = Gr** and can be efficiently computed by power iteration of the stochastic adjacency matrix (**G**) 
- Adding random uniform teleportation solves issues of dead-ends and spider-traps

## Random Walk with Restarts and Personalized PageRank

Given:

A bipartite graph representing user and item interactions (purchase)

Goal: Proximity on graphs

- What items should we recommend to a user who interacts with item Q?
- Intuition: if items Q and P are interacted by similar users, recommend P when user interacts with Q

<img src="https://raw.githubusercontent.com/zjwu0522/CS224W/main/images/image-20211010205151042.png" alt="image-20211010205151042" style="zoom:67%;" />

### Proximity on Graphs

<img src="https://raw.githubusercontent.com/zjwu0522/CS224W/main/images/image-20211010205232895.png" alt="image-20211010205232895" style="zoom: 80%;" />

### Random Walks

Idea

- Every node has some importance
- Importance gets evenly split among all edges and pushed to the neighbors

Given a set of *Query_nodes*, we simulate a random walk:

- Make a step to a random neighbor and record the visit (visit count)
- With probability *ALPHA*, restart the walk at one of the *Query_nodes*
- The nodes with the highest visit count have highest proximity to the *Query_node*

<img src="https://raw.githubusercontent.com/zjwu0522/CS224W/main/images/image-20211010205529192.png" alt="image-20211010205529192" style="zoom:67%;" />

Example result

<img src="https://raw.githubusercontent.com/zjwu0522/CS224W/main/images/image-20211010205650708.png" alt="image-20211010205650708" style="zoom:80%;" />

**Benefit**

The similarity considers:

1. Multiple connections
2. Multiple paths
3. Direct and indirect connections
4. Degree of the node

### Summary

<img src="https://raw.githubusercontent.com/zjwu0522/CS224W/main/images/image-20211010205816892.png" alt="image-20211010205816892" style="zoom:80%;" />

- A graph is naturally represented as a matrix
- We defined a random walk process over the graph
  - Random surfer moving across the links and with random teleportation
  - Stochastic adjacency matrix M
- PageRank = Limiting distribution of the surfer location represented node importance
  - Corresponds to the leading eigenvector of transformed adjacency matrix M

## Matrix Factorization and Node Embeddings

Can we connect random walk based network embedding to matrix factorization?

- Simplest **node similarity:** Nodes *u,v* are similar if they are connected by an edge
- This means **{z_T}_v · z_u = A_u,v**

### Matrix Factorization

- The embedding dimension *d* (number of rows in **Z**) is much smaller than number of nodes *n*
- Exact factorization  **A_u,v = {z_T}_v · z_u** is generally not possible
- However, we can learn **Z** approximately

<img src="https://raw.githubusercontent.com/zjwu0522/CS224W/main/images/image-20211010212425486.png" alt="image-20211010212425486" style="zoom:80%;" />

- Conclusion: inner product decoder with node similarity defined by edge connectivity is equivalent to **matrix factorization of A**

### Random Walk-based Similarity

- DeepWalk and node2vec have a more complex node similarity definition based on random walks
- **DeepWalk** is equivalent to matrix factorization of the following complex matrix expression:

<img src="https://raw.githubusercontent.com/zjwu0522/CS224W/main/images/image-20211010212746308.png" alt="image-20211010212746308" style="zoom: 80%;" />

- **Node2vec** can also be formulated as a matrix factorization 
- **Reference:** [Network Embedding as Matrix Factorization: Unifying DeepWalk, LINE, PTE, and node2vec](https://keg.cs.tsinghua.edu.cn/jietang/publications/WSDM18-Qiu-et-al-NetMF-network-embedding.pdf)

### Limitations

1. Cannot obtain embeddings for nodes not in the training set (inductive or transductive)

<img src="https://raw.githubusercontent.com/zjwu0522/CS224W/main/images/image-20211010213026999.png" alt="image-20211010213026999" style="zoom:67%;" />

2. Cannot capture **structural similarity**

<img src="https://raw.githubusercontent.com/zjwu0522/CS224W/main/images/image-20211010213107753.png" alt="image-20211010213107753" style="zoom:67%;" />

3. Cannot utilize node, edge and graph features

<img src="https://raw.githubusercontent.com/zjwu0522/CS224W/main/images/image-20211010213146651.png" alt="image-20211010213146651" style="zoom:67%;" />

## Conclusion

- **PageRank**
  - Measures importance of nodes in graph
  - Can be efficiently computed by power iteration of adjacency matrix
- **Personalized PageRank (PPR)**
  - Measures importance of nodes with respect to a particular node or set of nodes
  - Can be efficiently computed by random walk
- **Node embeddings** based on random walks can be expressed as **matrix factorization**
- **Viewing graphs as matrices plays a key role in all above algorithms!**
