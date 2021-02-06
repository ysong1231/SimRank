import copy
import time
import pytz
import pandas as pd
import numpy as np
import pickle as pkl
from datetime import datetime
from sklearn.metrics.pairwise import cosine_similarity

def update_progress(progress):
    barLength = 30
    status = ''
    if isinstance(progress, int):
        progress = float(progress)
    if not isinstance(progress, float):
        raise ValueError('Progress must be float')
    if progress < 0:
        raise ValueError('Progress below 0')
    if progress >= 1:
        progress = 1
        status = 'Done...\r\n'
    block = int(round(barLength * progress))
    text = f'\rPercent: [{"#" * block + "-" * (barLength - block)}] {round(progress * 100, 1)}% {status}'
    sys.stdout.write(text)
    sys.stdout.flush()
    
class SimRank(object):
    def __init__(self):
        self.Nodes = set()
        self.Graph = pd.DataFrame()
    
    def _create_graph(self, data, weighted, from_node_column, to_node_column, weight_column):
        self.Nodes = set(data[from_node_column].unique()) | set(data[to_node_column].unique())
        self.Graph = pd.DataFrame(np.zeros((len(self.Nodes), len(self.Nodes))), index = self.Nodes, columns = self.Nodes)
        if weighted:
            inNeighbors = data.groupby(to_node_column)[weight_column].sum().to_frame().rename(columns = {weight_column: 'inNeighbors'})  
        else:
            inNeighbors = data.groupby(to_node_column)[from_node_column].count().to_frame().rename(columns = {from_node_column: 'inNeighbors'})
        data = data.join(inNeighbors, on = in_node_column)
        data['_normailized_weight'] = (1.0 / data['inNeighbors']).replace([np.inf, -np.inf], np.nan).fillna(0)
        Graph = data.pivot(index = to_node_column, columns = from_node_column, values = '_normailized_weight').fillna(0)
        for _name, _row in Graph.iterrows():
            self.Graph.loc[_name][_row.index] = _row
    
    def _converged(self, s1, s2, eps):
        diff = (abs(s1 - s2) > eps).sum()
        if diff:
            return False
        return True
    
    def fit(self, data, C = 0.8, weighted = False, from_node_column = 'from', to_node_column = 'to', weight_column = 'weight', iterations = 100, eps = 1e-4, verbose = True):
        self._create_graph(data, weighted, from_node_column, to_node_column, weight_column)
        
        old_S = np.zeros((len(self.Nodes), len(self.Nodes)))
        new_S = np.zeros((len(self.Nodes), len(self.Nodes)))
        np.fill_diagonal(new_S, 1)
        if verbose:
            print('Start iterating...')
        for _iter in range(iterations):
            if self._converged(old_S, new_S, eps):
                if verbose:
                    print(f'\n\rConerged at iteration {_iter}, process complete!')
                break
            if verbose:
                update_progress(_iter/iterations)
            old_S = copy.deepcopy(new_S)
            new_S = C * self.Graph.values.dot(new_S).dot(self.Graph.T)
            np.fill_diagonal(new_S, 1)
        return pd.DataFrame(new_S, index = self.Nodes, columns = self.Nodes)
        
class BipartiteSimRank(object):
    def __init__(self):
        self.NodesGroup1 = set()
        self.NodesGroup2 = set()
        self.Graph_N1_N2 = pd.DataFrame()
        self.Graph_N2_N1 = pd.DataFrame()
    
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
    
    def _converged(self, s1, s2, eps):
        diff = (abs(s1 - s2) > eps).sum()
        if diff:
            return False
        return True
    
    def fit(self, data, C1 = 0.8, C2 = 0.8, weighted = False, node_group1_column = 'user', node_group2_column = 'item', weight_column = 'weight', iterations = 100, eps = 1e-4, verbose = True):
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
                    print(f'\n\rConerged at iteration {_iter}, process complete!')
                break
            if verbose:
                update_progress(_iter/iterations)
            old_S_N1 = copy.deepcopy(new_S_N1)
            new_S_N1 = C1 * self.Graph_N1_N2.values.dot(new_S_N2).dot(self.Graph_N1_N2.T)
            np.fill_diagonal(new_S_N1, 1)
            print(new_S_N1)
            old_S_N2 = copy.deepcopy(new_S_N2)
            new_S_N2 = C2 * self.Graph_N2_N1.values.dot(new_S_N1).dot(self.Graph_N2_N1.T)
            np.fill_diagonal(new_S_N2, 1)
        return pd.DataFrame(new_S_N1, index = self.NodesGroup1, columns = self.NodesGroup1), pd.DataFrame(new_S_N2, index = self.NodesGroup2, columns = self.NodesGroup2)

class weighted_bipartite_simrank:
    def __init__(self):
        self.N_user = None
        self.N_item = None
        self.users = None
        self.items = None
        self.S_user = None
        self.S_item = None
        self.G_rating = None

    def __str__(self):
        return f'Number of user: {self.N_user}\nNumber of item: {self.N_item}\nShape of user similarity matrix: {self.S_user.shape}\nShape of item similarity matrix: {self.S_item.shape}'

    def _create_graph_from_df(self, df):
        maxRatingByUser = df.groupby('userId')['rating'].sum().to_frame().rename(columns = {'rating': 'sumRatingByUser'})
        maxRatingByMovie = df.groupby('movieId')['rating'].sum().to_frame().rename(columns = {'rating': 'sumRatingByMovie'})

        df = df.join(maxRatingByUser, on = 'userId').join(maxRatingByMovie, on = 'movieId')
        df['norm_rating_by_user'] = df['rating'] / df['sumRatingByUser']
        df['norm_rating_by_movie'] = df['rating'] / df['sumRatingByMovie']

        self.G_rating = df.pivot(index = 'userId', columns = 'movieId', values = 'rating').fillna(0)
        G_user = df.pivot(index = 'userId', columns = 'movieId', values = 'norm_rating_by_user')
        G_item = df.pivot(index = 'movieId', columns = 'userId', values = 'norm_rating_by_movie')
        return G_user, G_item

    def _converged(self, sim1, sim2, eps):
        diff = (abs(sim1 - sim2) > eps).sum()
        if diff:
            return False
        return True
    
    def save(self, path = 'drive/MyDrive/Essay/', name = None):
        if name is None:
            tz_NY = pytz.timezone('America/New_York') 
            datetime_NY = datetime.now(tz_NY)
            name = f'WBS.{self.N_user}user{self.N_item}item.{datetime_NY.strftime("%Y-%m-%d-%H:%M:%S")}.pkl'
        with open(path + name, 'wb') as f:
            pkl.dump(self, f)
        print(f'Model saved as {path + name}')

    def _cal_W(self, G, N, obj, GPU = False):
        if obj == 'user':
            G_var = self.G_rating.replace(0, np.nan)
        else:
            G_var = self.G_rating.T.replace(0, np.nan)
        print(f'Initializing {obj}-W matrix...')
        start = time.time()
        spread = np.zeros((N, N))
        np.fill_diagonal(
            spread, 
            G_var.var(axis = 1).fillna(0).apply(lambda x: np.exp(-x))
            )
        if GPU:
            W = run_dot_product(spread, G.fillna(0))
        else:
            W = np.dot(spread, G.fillna(0))
        end = time.time()
        print(f"Finished in {end - start}s!")
        return W
    
    def _cal_E(self, G, obj, GPU = False):
        print(f"Initializing {obj} evidence matrix...")
        start = time.time()
        if GPU:
            E = run_dot_product((G > 0).astype(int), (G > 0).T.astype(int))
        else:
            E = np.dot((G > 0).astype(int), (G > 0).T.astype(int))
        E = 1 - 0.5 ** E
        end = time.time()
        print(f"Finished in {end - start}s!")
        return E

    def _cal_S(self, C_user, C_item, iterations, eps, GPU = False):
        S_item = np.zeros((self.N_item, self.N_item))
        np.fill_diagonal(S_item, 1)
        S_item_old = np.zeros((self.N_item, self.N_item))
        
        S_user = np.zeros((self.N_user, self.N_user))
        np.fill_diagonal(S_user, 1)
        S_user_old = np.zeros((self.N_user, self.N_user))

        for _iter in range(iterations):
            if self._converged(S_item, S_item_old, eps) and self._converged(S_user, S_user_old, eps):
                print(f"Converged at iteration {_iter}, break!")
                break
            
            print(f"Iteration {_iter + 1} / {iterations} start:")
            S_item_old = copy.deepcopy(S_item)
            S_user_old = copy.deepcopy(S_user)
            
            if GPU:
                # Try with GPU
                start1 = time.time()
                S_item = self.E_item * C_item * run_dot_product(run_dot_product(self.W_item_user, S_user), self.W_item_user.T)
            else:
                # Go with CPU
                start1 = time.time()
                S_item = self.E_item * C_item * np.dot(np.dot(self.W_item_user, S_user), self.W_item_user.T)
            np.fill_diagonal(S_item, 1)
            end1 = time.time()

            if GPU:
                # Try with GPU
                start2 = time.time()
                S_user = self.E_item * C_item * run_dot_product(run_dot_product(self.W_user_item, S_item), self.W_user_item.T)
            else:
                # Go with CPU
                start2 = time.time()
                print("GPU failed, trying with CPU...")
                S_user = self.E_user * C_user * np.dot(np.dot(self.W_user_item, S_item), self.W_user_item.T)
            np.fill_diagonal(S_user, 1)
            end2 = time.time()

            print(f"S_user updated in {end1 - start1}, S_item updated in {end2 - start2}!")

        return S_user, S_item
    
    def fit(self, df, C_item = 0.8, C_user = 0.8, iterations = 100, eps = 1e-4):
        G_user, G_item = self._create_graph_from_df(df)
        self.users = G_user.index
        self.items = G_item.index
        self.N_user = len(self.users)
        self.N_item = len(self.items)
        print(f"User count: {self.N_user}, item count: {self.N_item}")

        # Cal W for user and item
        self.W_user_item = self._cal_W(G_user, self.N_user, 'user')
        self.W_item_user = self._cal_W(G_item, self.N_item, 'item')
        
        # Cal E for user and item
        self.E_user = self._cal_E(G_user, 'user')
        self.E_item = self._cal_E(G_item, 'item')
        
        S_user, S_item = self._cal_S(C_user, C_item, iterations, eps, GPU = False)
        self.S_item = pd.DataFrame(S_item, index = self.items, columns = self.items)
        self.S_user = pd.DataFrame(S_user, index = self.users, columns = self.users)

    def compare_train_test_set(self, test_data):
        unique_train_user = set(self.users)
        unique_train_item = set(self.items)
        unique_test_user = set(test_data.userId.unique())
        unique_test_item = set(test_data.movieId.unique())
        cold_start_user = unique_test_user - (unique_test_user & unique_train_user)
        cold_start_item = unique_test_item - (unique_test_item & unique_train_item)
        print(f"Count of cold start user: {len(cold_start_user)}")
        print(f"Count of cold start item: {len(cold_start_item)}")
        return cold_start_user, cold_start_item

    def cf_recommendation(self, df, k = 10, cf_type = 'user', mask = True, melt = True):
        cold_start_user, cold_start_item = self.compare_train_test_set(df)
        user_to_predict = set(df.userId.unique()) - cold_start_user

        if cf_type == 'user':
            R = self.G_rating.replace(0, np.nan)
            mean_user_rating = R.mean(axis=1)
            diff = (R - mean_user_rating[:, np.newaxis]).fillna(0)

            pred = pd.DataFrame(
                user_to_predict, 
                columns = ['userId'])
            for item in self.items:
                pred[item] = np.nan
            pred = pred.set_index('userId')

            curr, total, poc = 0, len(user_to_predict), 0.1
            for user, _ in pred.iterrows():
                k_neighbors = set(self.S_user.loc[user].nlargest(k + 1).index) - set([user])
                s_neighbors = self.S_user[self.S_user.index.isin(k_neighbors)][user]
                s_neighbors = s_neighbors / s_neighbors.sum()
                diff_neighbors = diff[diff.index.isin(k_neighbors)]
                diag = np.zeros((k, k))
                np.fill_diagonal(diag, s_neighbors)
                weighted_diff = diag.dot(diff_neighbors).sum(axis = 0)
                pred.loc[user] = weighted_diff + mean_user_rating[user]
                if mask:
                    pred.loc[user] = pred.loc[user] * (self.G_rating.loc[user] == 0)
                curr += 1
                if curr / total >= poc:
                    print(f'{curr}/{total} completed')
                    poc += 0.1
            
        elif cf_type == 'item':
            item_to_recommend = self.items
            R = self.G_rating.replace(0, np.nan).T
            mean_item_rating = R.mean(axis = 1)
            diff = (R - mean_item_rating[:, np.newaxis]).fillna(0)[user_to_predict]

            pred = pd.DataFrame(
                item_to_recommend, 
                columns = ['movieId'])

            for user in user_to_predict:
                pred[user] = np.nan
            pred = pred.set_index('movieId')

            curr, total, poc = 0, len(item_to_recommend), 0.1
            for item, _ in pred.iterrows():
                k_neighbors = set(self.S_item.loc[item].nlargest(k + 1).index) - set([item])
                s_neighbors = self.S_item[self.S_item.index.isin(k_neighbors)][item]
                s_neighbors = s_neighbors / s_neighbors.sum()
                diff_neighbors = diff[diff.index.isin(k_neighbors)]
                diag = np.zeros((k, k))
                np.fill_diagonal(diag, s_neighbors)
                weighted_diff = diag.dot(diff_neighbors).sum(axis = 0)
                pred.loc[item] = weighted_diff + mean_item_rating[item]
                if mask:
                    pred.loc[item] = pred.loc[item] * (self.G_rating.T.loc[item] == 0)
                curr += 1
                if curr / total >= poc:
                    print(f'{curr}/{total} completed')
                    poc += 0.1  
        if melt:
            if cf_type == 'user':
                pred = pd.melt(pred.reset_index(), id_vars = ['userId'], value_vars = pred.columns).dropna()
                pred = pred[pred['value'] != 0].rename(columns = {'value': 'rating', 'variable': 'movieId'})
            else:
                pred = pd.melt(pred.reset_index(), id_vars = ['movieId'], value_vars = pred.columns).dropna()
                pred = pred[pred['value'] != 0].rename(columns = {'value': 'rating', 'variable': 'userId'})
        return pred

class tag_simrank:
    def __init__(self):
        self.N_user = None
        self.N_item = None
        self.users = None
        self.items = None
        self.S_user = None
        self.S_item = None
        self.G_rating = None
        self.G_tag = None
        self.S_tag_based = None

    def __str__(self):
        return f'Number of user: {self.N_user}\nNumber of item: {self.N_item}\nShape of user similarity matrix: {self.S_user.shape}\nShape of item similarity matrix: {self.S_item.shape}'

    def _create_graph_from_df(self, df):
        maxRatingByUser = df.groupby('userId')['rating'].sum().to_frame().rename(columns = {'rating': 'sumRatingByUser'})
        maxRatingByMovie = df.groupby('movieId')['rating'].sum().to_frame().rename(columns = {'rating': 'sumRatingByMovie'})

        df = df.join(maxRatingByUser, on = 'userId').join(maxRatingByMovie, on = 'movieId')
        df['norm_rating_by_user'] = df['rating'] / df['sumRatingByUser']
        df['norm_rating_by_movie'] = df['rating'] / df['sumRatingByMovie']

        self.G_rating = df.pivot(index = 'userId', columns = 'movieId', values = 'rating').fillna(0)
        G_user = df.pivot(index = 'userId', columns = 'movieId', values = 'norm_rating_by_user')
        G_item = df.pivot(index = 'movieId', columns = 'userId', values = 'norm_rating_by_movie')

        return G_user, G_item

    def _converged(self, sim1, sim2, eps):
        diff = (abs(sim1 - sim2) > eps).sum()
        if diff:
            return False
        return True
    
    def save(self, path = 'drive/MyDrive/Essay/', name = None):
        if name is None:
            tz_NY = pytz.timezone('America/New_York') 
            datetime_NY = datetime.now(tz_NY)
            name = f'WBS.{self.N_user}user{self.N_item}item.{datetime_NY.strftime("%Y-%m-%d-%H:%M:%S")}.pkl'
        with open(path + name, 'wb') as f:
            pkl.dump(self, f)
        print(f'Model saved as {path + name}')

    def _cal_tab_based_S(self, tag, how = 'jac', GPU = False):
        print("Initializing tab-based item similarity matrix...")
        start = time.time()
        tag = tag[tag.movieId.isin(self.items)]
        self.G_tag = tag.pivot(index = 'movieId', columns = 'tagId', values = 'relevance')
        if how == 'cos':
            S_tag_based = cosine_similarity(self.G_tag.fillna(0))
            
        elif how == 'jac':
            tag_or = pd.DataFrame(
                    np.zeros((self.N_item, self.N_item)),
                    index = self.items,
                    columns = self.items)
            for i, row in self.G_tag.iterrows():
                tag_or[i] = (self.G_tag + row).count(axis = 1)

            self.G_tag = self.G_tag.fillna(0)
            tag_and = np.dot(self.G_tag, self.G_tag.T)

            S_tag_based = (tag_and / tag_or).replace([np.inf, -np.inf], np.nan).fillna(0).values
        
        end = time.time()
        print(f"Finished in {end - start}s!")
        return S_tag_based

    def _cal_W(self, G, N, obj, GPU = False):
        if obj == 'user':
            G_var = self.G_rating
        else:
            G_var = self.G_rating.T
        print(f'Initializing {obj}-W matrix...')
        start = time.time()
        spread = np.zeros((N, N))
        np.fill_diagonal(
            spread, 
            G_var.var(axis = 1).fillna(0).apply(lambda x: np.exp(-x))
            )
        if GPU:
            W = run_dot_product(spread, G.fillna(0))
        else:
            W = np.dot(spread, G.fillna(0))
        end = time.time()
        print(f"Finished in {end - start}s!")
        return W
    
    def _cal_E(self, G, obj, GPU = False):
        print(f"Initializing {obj} evidence matrix...")
        start = time.time()
        if GPU:
            E = run_dot_product((G > 0).astype(int), (G > 0).T.astype(int))
        else:
            E = np.dot((G > 0).astype(int), (G > 0).T.astype(int))
        E = 1 - 0.5 ** E
        end = time.time()
        print(f"Finished in {end - start}s!")
        return E

    def _cal_S(self, C_user, C_item, lbd, iterations, eps, GPU = False):
        S_item = np.zeros((self.N_item, self.N_item))
        np.fill_diagonal(S_item, 1)
        S_item_old = np.zeros((self.N_item, self.N_item))
        
        S_user = np.zeros((self.N_user, self.N_user))
        np.fill_diagonal(S_user, 1)
        S_user_old = np.zeros((self.N_user, self.N_user))

        for _iter in range(iterations):
            if self._converged(S_item, S_item_old, eps) and self._converged(S_user, S_user_old, eps):
                print(f"Converged at iteration {_iter}, break!")
                break
            
            print(f"Iteration {_iter + 1} / {iterations} start:")
            S_item_old = copy.deepcopy(S_item)
            S_user_old = copy.deepcopy(S_user)
            
            if GPU:
                # Try with GPU
                start1 = time.time()
                S_item = self.E_item * C_item * run_dot_product(run_dot_product(self.W_item_user, S_user), self.W_item_user.T)
            else:
                # Go with CPU
                start1 = time.time()
                S_item = self.E_item * C_item * np.dot(np.dot(self.W_item_user, S_user), self.W_item_user.T)

            S_item = (1 - lbd) * S_item + lbd * self.S_tag_based
            np.fill_diagonal(S_item, 1)
            end1 = time.time()

            if GPU:
                # Try with GPU
                start2 = time.time()
                S_user = self.E_item * C_item * run_dot_product(run_dot_product(self.W_user_item, S_item), self.W_user_item.T)
            else:
                # Go with CPU
                start2 = time.time()
                print("GPU failed, trying with CPU...")
                S_user = self.E_user * C_user * np.dot(np.dot(self.W_user_item, S_item), self.W_user_item.T)
            np.fill_diagonal(S_user, 1)
            end2 = time.time()
            print(f"S_user updated in {end1 - start1}, S_item updated in {end2 - start2}!")

        return S_user, S_item

    def fit(self, df, tag, C_item = 0.8, C_user = 0.8, lbd = 0.3, iterations = 100, eps = 1e-4, how = 'cos'):
        G_user, G_item = self._create_graph_from_df(df)
        self.users = G_user.index
        self.items = G_item.index
        self.N_user = len(self.users)
        self.N_item = len(self.items)
        print(f"User count: {self.N_user}, item count: {self.N_item}")
        
        self.S_tag_based = self._cal_tab_based_S(tag, how = 'jac', GPU = False)

        self.W_user_item = self._cal_W(G_user, self.N_user, 'user', GPU = False)
        self.W_item_user = self._cal_W(G_item, self.N_item, 'item', GPU = False)
        
        self.E_user = self._cal_E(G_user, 'user', GPU = False)
        self.E_item = self._cal_E(G_item, 'item', GPU = False)
        
        S_user, S_item = self._cal_S(C_user, C_item, lbd, iterations, eps, GPU = False)

        self.S_item = pd.DataFrame(S_item, index = self.items, columns = self.items)
        self.S_user = pd.DataFrame(S_user, index = self.users, columns = self.users)

    def compare_train_test_set(self, test_data):
        unique_train_user = set(self.users)
        unique_train_item = set(self.items)
        unique_test_user = set(test_data.userId.unique())
        unique_test_item = set(test_data.movieId.unique())
        cold_start_user = unique_test_user - (unique_test_user & unique_train_user)
        cold_start_item = unique_test_item - (unique_test_item & unique_train_item)
        print(f"Count of cold start user: {len(cold_start_user)}")
        print(f"Count of cold start item: {len(cold_start_item)}")
        return cold_start_user, cold_start_item

    def cf_recommendation(self, df, k = 10, cf_type = 'user', mask = True, melt = True):
        cold_start_user, cold_start_item = self.compare_train_test_set(df)
        user_to_predict = set(df.userId.unique()) - cold_start_user

        if cf_type == 'user':
            R = self.G_rating.replace(0, np.nan)
            mean_user_rating = R.mean(axis=1)
            diff = (R - mean_user_rating[:, np.newaxis]).fillna(0)

            pred = pd.DataFrame(
                user_to_predict, 
                columns = ['userId'])
            for item in self.items:
                pred[item] = np.nan
            pred = pred.set_index('userId')

            curr, total, poc = 0, len(user_to_predict), 0.1
            for user, _ in pred.iterrows():
                k_neighbors = set(self.S_user.loc[user].nlargest(k + 1).index) - set([user])
                s_neighbors = self.S_user[self.S_user.index.isin(k_neighbors)][user]
                s_neighbors = s_neighbors / s_neighbors.sum()
                diff_neighbors = diff[diff.index.isin(k_neighbors)]
                diag = np.zeros((k, k))
                np.fill_diagonal(diag, s_neighbors)
                weighted_diff = diag.dot(diff_neighbors).sum(axis = 0)
                pred.loc[user] = weighted_diff + mean_user_rating[user]
                if mask:
                    pred.loc[user] = pred.loc[user] * (self.G_rating.loc[user] == 0)
                curr += 1
                if curr / total >= poc:
                    print(f'{curr}/{total} completed')
                    poc += 0.1
            
        elif cf_type == 'item':
            item_to_recommend = self.items
            R = self.G_rating.replace(0, np.nan).T
            mean_item_rating = R.mean(axis = 1)
            diff = (R - mean_item_rating[:, np.newaxis]).fillna(0)[user_to_predict]

            pred = pd.DataFrame(
                item_to_recommend, 
                columns = ['movieId'])

            for user in user_to_predict:
                pred[user] = np.nan
            pred = pred.set_index('movieId')

            curr, total, poc = 0, len(item_to_recommend), 0.1
            for item, _ in pred.iterrows():
                k_neighbors = set(self.S_item.loc[item].nlargest(k + 1).index) - set([item])
                s_neighbors = self.S_item[self.S_item.index.isin(k_neighbors)][item]
                s_neighbors = s_neighbors / s_neighbors.sum()
                diff_neighbors = diff[diff.index.isin(k_neighbors)]
                diag = np.zeros((k, k))
                np.fill_diagonal(diag, s_neighbors)
                weighted_diff = diag.dot(diff_neighbors).sum(axis = 0)
                pred.loc[item] = weighted_diff + mean_item_rating[item]
                if mask:
                    pred.loc[item] = pred.loc[item] * (self.G_rating.T.loc[item] == 0)
                curr += 1
                if curr / total >= poc:
                    print(f'{curr}/{total} completed')
                    poc += 0.1  
        if melt:
            if cf_type == 'user':
                pred = pd.melt(pred.reset_index(), id_vars = ['userId'], value_vars = pred.columns).dropna()
                pred = pred[pred['value'] != 0].rename(columns = {'value': 'rating', 'variable': 'movieId'})
            else:
                pred = pd.melt(pred.reset_index(), id_vars = ['movieId'], value_vars = pred.columns).dropna()
                pred = pred[pred['value'] != 0].rename(columns = {'value': 'rating', 'variable': 'userId'})
        return pred