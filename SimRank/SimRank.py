import sys
import copy
import time
import pandas as pd
import numpy as np
from SimRank.Helper import *
    
class SimRank(object):
    """
    Basic class of the SimRank algorithm. Calculate the similarities between the
    nodes in a weighted/unweighted directive graph.

    Attributes
    ----------
    Nodes: set
        A set of string representing the nodes in the given graph.
    Graph: pandas.DataFrame
        A padas DataFrame representing the graph
    """
    def __init__(self):
        self.Nodes = set()
        self.Graph = pd.DataFrame()
    
    def _create_graph(self, data, weighted, from_node_column, to_node_column, weight_column):
        """
        Create a graph from the given data frame by using the given columns 
        and pivot the given dataframe.

        Parameters
        ----------
        data : pandas.DataFrame
            Raw data.
        weighted : boolean
            Whether considering weights when pivoting the raw data.
        from_node_column : string
            The column name of the starting nodes.
        to_node_column : string
            The column name of the ending nodes.
        weight_column : string
            The column name of the weightings.
        """
        self.Nodes = set(data[from_node_column].unique()) | set(data[to_node_column].unique())
        self.Graph = pd.DataFrame(np.zeros((len(self.Nodes), len(self.Nodes))), index = self.Nodes, columns = self.Nodes)
        if weighted:
            inNeighbors = data.groupby(to_node_column)[weight_column].sum().to_frame().rename(columns = {weight_column: 'inNeighbors'})  
        else:
            inNeighbors = data.groupby(to_node_column)[from_node_column].count().to_frame().rename(columns = {from_node_column: 'inNeighbors'})
        data = data.join(inNeighbors, on = to_node_column)
        data['_normailized_weight'] = (1.0 / data['inNeighbors']).replace([np.inf, -np.inf], np.nan).fillna(0)
        Graph = data.pivot(index = to_node_column, columns = from_node_column, values = '_normailized_weight').fillna(0)
        for _name, _row in Graph.iterrows():
            self.Graph.loc[_name][_row.index] = _row
    
    def _converged(self, s1, s2, eps):
        """
        A helper function to indicate whether the difference between the two given
        matrixs are smaller than the given error.

        Parameters
        ----------
        s1 : numpy.array
            Matrix 1 for comparation.
        s2 : numpy.array
            Matrix 1 for comparation. Must be the same shape as s1.
        eps : float
            The error threshold.
        
        Returns
        ----------
        Boolean
            True, if s1 and s2's difference are smaller than the given error;
            False, if not
        """
        diff = (abs(s1 - s2) > eps).sum()
        if diff:
            return False
        return True
    
    def fit(self, data, C = 0.8, weighted = False, from_node_column = 'from', to_node_column = 'to', 
            weight_column = 'weight', iterations = 100, eps = 1e-4, verbose = True):
        """
        The main method to run the recursion iterations. Calculating the 
        similarities between nodes from the input data. Return a similarities 
        matrix.

        Parameters
        ----------
        data : pandas.DataFrame
            Raw data.
        C : float, optional
            Damping coefficient defined in the algorithm. 
            The default value is 0.8.
        weighted : boolean, optional
            Whether considering weights when pivoting the raw data.
            The default value is False.
        from_node_column : string, optional
            The column name of the starting nodes.
            The default column name is "from".
        to_node_column : string, optional
            The column name of the ending nodes.
            The default column name is "to".
        weight_column : string, optional
            The column name of the weightings.
            The default column name is "weight".
        iterations : integer, optional
            The max iteration number of the calculation process. The 
            process is end when reached the max iteration times.
            The default max iteration time is 100.
        eps : float, optional
            The error threshold. The iteration will be stopped when the 
            difference is less than the given error.
            The default number is 1e-4.
        verbose : boolean, optional
            Deciding whether the intermediate progress will be printed out.
            The intermediate progress will be printed out by default.

        Returns
        ----------
        pandas.DataFrame
            A pandas DataFrame storing the similarities between the nodes
            in the given graph.
        """
        self._create_graph(data, weighted, from_node_column, to_node_column, weight_column)
        old_S = np.zeros((len(self.Nodes), len(self.Nodes)))
        new_S = np.zeros((len(self.Nodes), len(self.Nodes)))
        np.fill_diagonal(new_S, 1)
        if verbose:
            print('Start iterating...')
        for _iter in range(iterations):
            if self._converged(old_S, new_S, eps):
                if verbose:
                    text = f'\rPercent: [{"#" * BAR_LENGTH}] 100% Complete! \n\rConverged at iteration {_iter}'
                    sys.stdout.write(text)
                    sys.stdout.flush()
                break
            if verbose:
                update_progress(_iter/iterations)
            old_S = copy.deepcopy(new_S)
            new_S = C * self.Graph.values.dot(new_S).dot(self.Graph.T)
            np.fill_diagonal(new_S, 1)
        return pd.DataFrame(new_S, index = self.Nodes, columns = self.Nodes)
        
class BipartiteSimRank(object):
    """
    Basic class of the SimRank algorithm on a bipartitle graph. Calculate the 
    similarities between the two group of nodes in a weighted/unweighted bipartitle 
    graph separately.

    Attributes
    ----------
    NodesGroup1: set
        A set of string representing the first group of nodes in the given graph.
    NodesGroup2: set
        A set of string representing the second group of nodes in the given graph.
    Graph_N1_N2: pandas.DataFrame
        A padas DataFrame representing the graph from the first group of nodes to 
        the second group of nodes.
    Graph_N2_N1: pandas.DataFrame
        A padas DataFrame representing the graph from the second group of nodes to 
        the first group of nodes.
    """
    def __init__(self):
        self.NodesGroup1 = set()
        self.NodesGroup2 = set()
        self.Graph_N1_N2 = pd.DataFrame()
        self.Graph_N2_N1 = pd.DataFrame()
    
    def _create_graph(self, data, weighted, node_group1_column, node_group2_column, weight_column):
        """
        Create a graph from the given data frame by using the given columns 
        and pivot the given dataframe.

        Parameters
        ----------
        data : pandas.DataFrame
            Raw data.
        weighted : boolean
            Whether considering weights when pivoting the raw data.
        node_group1_column : string
            The column name of the first group of nodes.
        node_group2_column : string
            The column name of the second group of nodes.
        weight_column : string
            The column name of the weightings.
        """
        self.NodesGroup1 = set(data[node_group1_column].unique())
        self.NodesGroup2 = set(data[node_group2_column].unique())
        self.Graph_N1_N2 = pd.DataFrame(np.zeros((len(self.NodesGroup1), len(self.NodesGroup2))), index = self.NodesGroup1, columns = self.NodesGroup2)
        self.Graph_N2_N1 = pd.DataFrame(np.zeros((len(self.NodesGroup2), len(self.NodesGroup1))), index = self.NodesGroup2, columns = self.NodesGroup1)
        if weighted:
            NodeGroup1Neighbors = data.groupby(node_group1_column)[weight_column].sum().to_frame().rename(columns = {weight_column: 'node_group1_neighbors'})
            NodeGroup2Neighbors = data.groupby(node_group2_column)[weight_column].sum().to_frame().rename(columns = {weight_column: 'node_group2_neighbors'})
        else:
            NodeGroup1Neighbors = data.groupby(node_group1_column)[node_group2_column].count().to_frame().rename(columns = {node_group2_column: 'node_group1_neighbors'})
            NodeGroup2Neighbors = data.groupby(node_group2_column)[node_group1_column].count().to_frame().rename(columns = {node_group1_column: 'node_group2_neighbors'})
        data = data.join(NodeGroup1Neighbors, on = node_group1_column).join(NodeGroup2Neighbors, on = node_group2_column)
        data['n1_n2_normailized_weight'] = (1.0 / data['node_group1_neighbors']).replace([np.inf, -np.inf], np.nan).fillna(0)
        data['n2_n1_normailized_weight'] = (1.0 / data['node_group2_neighbors']).replace([np.inf, -np.inf], np.nan).fillna(0)
        self.Graph_N1_N2 = data.pivot(index = node_group1_column, columns = node_group2_column, values = 'n1_n2_normailized_weight').fillna(0)
        self.Graph_N2_N1 = data.pivot(index = node_group2_column, columns = node_group1_column, values = 'n2_n1_normailized_weight').fillna(0)
    
    def _converged(self, s1, s2, eps):
        """
        A helper function to indicate whether the difference between the two given
        matrixs are smaller than the given error.

        Parameters
        ----------
        s1 : numpy.array
            Matrix 1 for comparation.
        s2 : numpy.array
            Matrix 1 for comparation. Must be the same shape as s1.
        eps : float
            The error threshold.
        
        Returns
        ----------
        Boolean
            True, if s1 and s2's difference are smaller than the given error;
            False, if not
        """
        diff = (abs(s1 - s2) > eps).sum()
        if diff:
            return False
        return True
    
    def fit(self, data, C1 = 0.8, C2 = 0.8, weighted = False, node_group1_column = 'user', node_group2_column = 'item', 
            weight_column = 'weight', iterations = 100, eps = 1e-4, verbose = True):
        """
        The main method to run the recursion iterations. Calculating the 
        similarities between nodes from the input data. Return a similarities 
        matrix.

        Parameters
        ----------
        data : pandas.DataFrame
            Raw data.
        C1 : float, optional
            Damping coefficient defined in the algorithm that will be multiplied
            to the first group of nodes' similarities during the iteration.
            The default value is 0.8.
        C2 : float, optional
            Damping coefficient defined in the algorithm that will be multiplied
            to the second group of nodes' similarities during the iteration.
            The default value is 0.8.
        weighted : boolean, optional
            Whether considering weights when pivoting the raw data.
            The default value is False.
        node_group1_column : string, optional
            The column name of the first group of nodes.
            The default column name is "user".
        node_group2_column : string, optional
            The column name of the second group of nodes.
            The default column name is "item".
        weight_column : string, optional
            The column name of the weightings.
            The default column name is "weight".
        iterations : integer, optional
            The max iteration number of the calculation process. The 
            process is end when reached the max iteration times.
            The default max iteration time is 100.
        eps : float, optional
            The error threshold. The iteration will be stopped when the 
            difference is less than the given error.
            The default number is 1e-4.
        verbose : boolean, optional
            Deciding whether the intermediate progress will be printed out.
            The intermediate progress will be printed out by default.

        Returns
        ----------
        pandas.DataFrame
            A pandas DataFrame storing the similarities between the first
            group of nodes in the given graph.
        pandas.DataFrame
            A pandas DataFrame storing the similarities between the second
            group of nodes in the given graph.
        """
        self._create_graph(data, weighted, node_group1_column, node_group2_column, weight_column)
        old_S_N1 = np.zeros((len(self.NodesGroup1), len(self.NodesGroup1)))
        new_S_N1 = np.zeros((len(self.NodesGroup1), len(self.NodesGroup1)))
        np.fill_diagonal(new_S_N1, 1)
        old_S_N2 = np.zeros((len(self.NodesGroup2), len(self.NodesGroup2)))
        new_S_N2 = np.zeros((len(self.NodesGroup2), len(self.NodesGroup2)))
        np.fill_diagonal(new_S_N2, 1)
        if verbose:
            print('Start iterating...')
        for _iter in range(iterations):
            if self._converged(old_S_N1, new_S_N1, eps) and self._converged(old_S_N2, new_S_N2, eps):
                if verbose:
                    text = f'\rPercent: [{"#" * BAR_LENGTH}] 100% Complete! \n\rConverged at iteration {_iter}'
                    sys.stdout.write(text)
                    sys.stdout.flush()
                break
            if verbose:
                update_progress(_iter/iterations)
            old_S_N1 = copy.deepcopy(new_S_N1)
            new_S_N1 = C1 * self.Graph_N1_N2.values.dot(new_S_N2).dot(self.Graph_N1_N2.T)
            np.fill_diagonal(new_S_N1, 1)
            old_S_N2 = copy.deepcopy(new_S_N2)
            new_S_N2 = C2 * self.Graph_N2_N1.values.dot(new_S_N1).dot(self.Graph_N2_N1.T)
            np.fill_diagonal(new_S_N2, 1)
        return pd.DataFrame(new_S_N1, index = self.NodesGroup1, columns = self.NodesGroup1), pd.DataFrame(new_S_N2, index = self.NodesGroup2, columns = self.NodesGroup2)

class SimRankPP(SimRank):
    def __init__(self):
        super(SimRankPP, self).__init__()
        self.Evidence = pd.DataFrame()
        self.Weight = pd.DataFrame()
        
    def _cal_Evidence(self, G, verbose):
        if verbose:
            print("Initializing Evidence matrix...")
        start = time.time()
        E = np.dot((G > 0).astype(int), (G > 0).T.astype(int))
        E = 1 - 0.5 ** E
        end = time.time()
        if verbose:
            print(f"Finished in {end - start}s!")
        return E
    
    def _cal_Weight(self, G, verbose):
        if verbose:
            print(f'Initializing Weight matrix...')
        start = time.time()
        G_var = G.replace(0, np.nan)
        N = len(G)
        spread = np.zeros((N, N))
        np.fill_diagonal(
            spread, 
            G_var.var(axis = 1).fillna(0).apply(lambda x: np.exp(-x))
            )
        W = np.dot(spread, G.fillna(0))
        end = time.time()
        if verbose:
            print(f"Finished in {end - start}s!")
        return W
    
    def fit(self, data, C = 0.8, weighted = False, from_node_column = 'from', to_node_column = 'to', weight_column = 'weight', iterations = 100, eps = 1e-4, verbose = True):
        self._create_graph(data, weighted, from_node_column, to_node_column, weight_column)
        # Cal W for user and item
        self.Weight = self._cal_Weight(self.Graph, verbose)
        # Cal E for user and item
        self.Evidence = self._cal_Evidence(self.Graph, verbose)
        # Init Similarity Matrix
        old_S = np.zeros((len(self.Nodes), len(self.Nodes)))
        new_S = np.zeros((len(self.Nodes), len(self.Nodes)))
        np.fill_diagonal(new_S, 1)
        if verbose:
            print('Start iterating...')
        for _iter in range(iterations):
            if self._converged(old_S, new_S, eps):
                if verbose:
                    text = f'\rPercent: [{"#" * BAR_LENGTH}] 100% Complete! \n\rConverged at iteration {_iter}'
                    sys.stdout.write(text)
                    sys.stdout.flush()
                break
            if verbose:
                update_progress(_iter/iterations)
            old_S = copy.deepcopy(new_S)
            new_S = self.Evidence * C * self.Weight.dot(new_S).dot(self.Weight.T)
            np.fill_diagonal(new_S, 1)
        return pd.DataFrame(new_S, index = self.Nodes, columns = self.Nodes)

class BipartiteSimRankPP(SimRankPP):
    def __init__(self):
        self.NodesGroup1 = set()
        self.NodesGroup2 = set()
        self.Graph_N1_N2 = pd.DataFrame()
        self.Graph_N2_N1 = pd.DataFrame()
        self.Evidence_N1 = pd.DataFrame()
        self.Evidence_N2 = pd.DataFrame()
        self.Weight_N1 = pd.DataFrame()
        self.Weight_N2 = pd.DataFrame()
    
    def _create_graph(self, data, weighted, node_group1_column, node_group2_column, weight_column):
        self.NodesGroup1 = set(data[node_group1_column].unique())
        self.NodesGroup2 = set(data[node_group2_column].unique())
        self.Graph_N1_N2 = pd.DataFrame(np.zeros((len(self.NodesGroup1), len(self.NodesGroup2))), index = self.NodesGroup1, columns = self.NodesGroup2)
        self.Graph_N2_N1 = pd.DataFrame(np.zeros((len(self.NodesGroup2), len(self.NodesGroup1))), index = self.NodesGroup2, columns = self.NodesGroup1)
        if weighted:
            NodeGroup1Neighbors = data.groupby(node_group1_column)[weight_column].sum().to_frame().rename(columns = {weight_column: 'node_group1_neighbors'})
            NodeGroup2Neighbors = data.groupby(node_group2_column)[weight_column].sum().to_frame().rename(columns = {weight_column: 'node_group2_neighbors'})
        else:
            NodeGroup1Neighbors = data.groupby(node_group1_column)[node_group2_column].count().to_frame().rename(columns = {node_group2_column: 'node_group1_neighbors'})
            NodeGroup2Neighbors = data.groupby(node_group2_column)[node_group1_column].count().to_frame().rename(columns = {node_group1_column: 'node_group2_neighbors'})
        data = data.join(NodeGroup1Neighbors, on = node_group1_column).join(NodeGroup2Neighbors, on = node_group2_column)
        data['n1_n2_normailized_weight'] = (1.0 / data['node_group1_neighbors']).replace([np.inf, -np.inf], np.nan).fillna(0)
        data['n2_n1_normailized_weight'] = (1.0 / data['node_group2_neighbors']).replace([np.inf, -np.inf], np.nan).fillna(0)
        self.Graph_N1_N2 = data.pivot(index = node_group1_column, columns = node_group2_column, values = 'n1_n2_normailized_weight').fillna(0)
        self.Graph_N2_N1 = data.pivot(index = node_group2_column, columns = node_group1_column, values = 'n2_n1_normailized_weight').fillna(0)
    
    def fit(self, data, C1 = 0.8, C2 = 0.8, weighted = False, node_group1_column = 'user', node_group2_column = 'item', weight_column = 'weight', iterations = 100, eps = 1e-4, verbose = True):
        self._create_graph(data, weighted, node_group1_column, node_group2_column, weight_column)
        # Cal W for user and item
        self.Weight_N1 = self._cal_Weight(self.Graph_N1_N2, verbose)
        self.Weight_N2 = self._cal_Weight(self.Graph_N2_N1, verbose)
        # Cal E for user and item
        self.Evidence_N1 = self._cal_Evidence(self.Graph_N1_N2, verbose)
        self.Evidence_N2 = self._cal_Evidence(self.Graph_N2_N1, verbose)
        # Init Similarity Matrix
        old_S_N1 = np.zeros((len(self.NodesGroup1), len(self.NodesGroup1)))
        new_S_N1 = np.zeros((len(self.NodesGroup1), len(self.NodesGroup1)))
        np.fill_diagonal(new_S_N1, 1)
        old_S_N2 = np.zeros((len(self.NodesGroup2), len(self.NodesGroup2)))
        new_S_N2 = np.zeros((len(self.NodesGroup2), len(self.NodesGroup2)))
        np.fill_diagonal(new_S_N2, 1)
        if verbose:
            print('Start iterating...')
        for _iter in range(iterations):
            if self._converged(old_S_N1, new_S_N1, eps) and self._converged(old_S_N2, new_S_N2, eps):
                if verbose:
                    text = f'\rPercent: [{"#" * BAR_LENGTH}] 100% Complete! \n\rConverged at iteration {_iter}'
                    sys.stdout.write(text)
                    sys.stdout.flush()
                break
            if verbose:
                update_progress(_iter/iterations)
            old_S_N1 = copy.deepcopy(new_S_N1)
            new_S_N1 = self.Evidence_N1 * C1 * self.Weight_N1.dot(new_S_N2).dot(self.Weight_N1.T)
            np.fill_diagonal(new_S_N1, 1)
            old_S_N2 = copy.deepcopy(new_S_N2)
            new_S_N2 = self.Evidence_N1 * C2 * self.Weight_N2.dot(new_S_N1).dot(self.Weight_N2.T)
            np.fill_diagonal(new_S_N2, 1)
        return pd.DataFrame(new_S_N1, index = self.NodesGroup1, columns = self.NodesGroup1), pd.DataFrame(new_S_N2, index = self.NodesGroup2, columns = self.NodesGroup2)

class AprioriSimRank(SimRankPP):
    def __init__(self):
        super(AprioriSimRank, self).__init__()
    
    def fit(self, data, AprioriSim, C = 0.8, lbd = 0.5, weighted = False, from_node_column = 'from', to_node_column = 'to', weight_column = 'weight', iterations = 100, eps = 1e-4, verbose = True):
        self._create_graph(data, weighted, from_node_column, to_node_column, weight_column)
        # Cal W for user and item
        self.Weight = self._cal_Weight(self.Graph, verbose)
        # Cal E for user and item
        self.Evidence = self._cal_Evidence(self.Graph, verbose)
        # Init Similarity Matrix
        old_S = np.zeros((len(self.Nodes), len(self.Nodes)))
        new_S = np.zeros((len(self.Nodes), len(self.Nodes)))
        np.fill_diagonal(new_S, 1)
        if verbose:
            print('Start iterating...')
        for _iter in range(iterations):
            if self._converged(old_S, new_S, eps):
                if verbose:
                    text = f'\rPercent: [{"#" * BAR_LENGTH}] 100% Complete! \n\rConverged at iteration {_iter}'
                    sys.stdout.write(text)
                    sys.stdout.flush()
                break
            if verbose:
                update_progress(_iter/iterations)
            old_S = copy.deepcopy(new_S)
            new_S = (1 - lbd) * self.Evidence * C * self.Weight.dot(new_S).dot(self.Weight.T) + lbd * AprioriSim
            np.fill_diagonal(new_S, 1)
        return pd.DataFrame(new_S, index = self.Nodes, columns = self.Nodes)

class BipartitleAprioriSimRank(BipartiteSimRankPP):
    def __init__(self):
        super(BipartitleAprioriSimRank, self).__init__()
    
    def fit(self, data, AprioriSim1, AprioriSim2, C1 = 0.8, C2 = 0.8, lbd1 = 0.5, lbd2 = 0.5, weighted = False, node_group1_column = 'user', node_group2_column = 'item', weight_column = 'weight', iterations = 100, eps = 1e-4, verbose = True):
        self._create_graph(data, weighted, node_group1_column, node_group2_column, weight_column)
        # Cal W for user and item
        self.Weight_N1 = self._cal_Weight(self.Graph_N1_N2, verbose)
        self.Weight_N2 = self._cal_Weight(self.Graph_N2_N1, verbose)
        # Cal E for user and item
        self.Evidence_N1 = self._cal_Evidence(self.Graph_N1_N2, verbose)
        self.Evidence_N2 = self._cal_Evidence(self.Graph_N2_N1, verbose)
        # Init Similarity Matrix
        old_S_N1 = np.zeros((len(self.NodesGroup1), len(self.NodesGroup1)))
        new_S_N1 = np.zeros((len(self.NodesGroup1), len(self.NodesGroup1)))
        np.fill_diagonal(new_S_N1, 1)
        old_S_N2 = np.zeros((len(self.NodesGroup2), len(self.NodesGroup2)))
        new_S_N2 = np.zeros((len(self.NodesGroup2), len(self.NodesGroup2)))
        np.fill_diagonal(new_S_N2, 1)
        if verbose:
            print('Start iterating...')
        for _iter in range(iterations):
            if self._converged(old_S_N1, new_S_N1, eps) and self._converged(old_S_N2, new_S_N2, eps):
                if verbose:
                    text = f'\rPercent: [{"#" * BAR_LENGTH}] 100% Complete! \n\rConverged at iteration {_iter}'
                    sys.stdout.write(text)
                    sys.stdout.flush()
                break
            if verbose:
                update_progress(_iter/iterations)
            old_S_N1 = copy.deepcopy(new_S_N1)
            new_S_N1 = (1 - lbd1) * self.Evidence_N1 * C1 * self.Weight_N1.dot(new_S_N2).dot(self.Weight_N1.T) + lbd1 * AprioriSim1
            np.fill_diagonal(new_S_N1, 1)
            old_S_N2 = copy.deepcopy(new_S_N2)
            new_S_N2 = (1 - lbd2) * self.Evidence_N1 * C2 * self.Weight_N2.dot(new_S_N1).dot(self.Weight_N2.T) + lbd2 * AprioriSim2
            np.fill_diagonal(new_S_N2, 1)
        return pd.DataFrame(new_S_N1, index = self.NodesGroup1, columns = self.NodesGroup1), pd.DataFrame(new_S_N2, index = self.NodesGroup2, columns = self.NodesGroup2)