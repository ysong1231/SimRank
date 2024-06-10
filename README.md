# SimRank
A Python implementation of SimRank and SimRank++ algorithms on both directive graphs and bipartitle graphs, with matrix manipulations to reduce the calculation complexity effectively.

* The algorithm is implemented in a matrix manipulation fashion instead of the traditional recursion method. 

### Author
Yilin Song

### Requirements
1. Pandas
2. Numpy

### Examples
This repo includes simple examples using the [BTS Flight](https://www.transtats.bts.gov/DL_SelectFields.asp?Table_ID=236) dataset to demonstrate the usage of **SimRank** class (SimRank algorithm) and **SimRankPP** class (SimRank++ algorithm) on a directive graph.

Also, the [MovieLens](https://grouplens.org/datasets/movielens/) dataset was used for demonstrating the usage of **BipartitleSimRank** class (SimRank algorithm) and **BipartitleSimRankPP** class (SimRank++ algorithm) on a bipartitel graph.

Check the [sample notebook](https://github.com/ysong1231/SimRank/blob/main/examples/basic_examples.ipynb) for details
