import copy
import time
import pytz
import pandas as pd
import numpy as np
import pickle as pkl
from datetime import datetime
from sklearn.metrics.pairwise import cosine_similarity
#from GPU_helper import run_dot_product

class naive_bipartite_simrank:
    def __init__(self):
        self.N_user = None
        self.N_item = None
        self.users = None
        self.items = None
        self.S_user = None
        self.S_item = None
        self.Rating = None

    def __str__(self):
        return f'Number of user: {self.N_user}\nNumber of item: {self.N_item}\nShape of user similarity matrix: {self.S_user.shape}\nShape of item similarity matrix: {self.S_item.shape}'

    def _create_graph_from_df(self, df):
        self.Rating = df.pivot(index = 'userId', columns = 'movieId', values = 'rating').fillna(0)

    def _converged(self, sim1, sim2, eps):
        diff = (abs(sim1 - sim2) > eps).sum()
        if diff:
            return False
        return True
    
    def save(self, path = 'drive/MyDrive/Essay/', name = None):
        if name is None:
            tz_NY = pytz.timezone('America/New_York') 
            datetime_NY = datetime.now(tz_NY)
            name = f'NBS.{self.N_user}user{self.N_item}item.{datetime_NY.strftime("%Y-%m-%d-%H:%M:%S")}.pkl'
        with open(path + name, 'wb') as f:
            pkl.dump(self, f)
        print(f'Model saved as {path + name}')

    def fit(self, df, C_item = 0.8, C_user = 0.8, iterations = 100, eps = 1e-4):
        self._create_graph_from_df(df)
        self.users = self.Rating.index
        self.items = self.Rating.columns
        self.N_user = len(self.users)
        self.N_item = len(self.items)
        print(f"User count: {self.N_user}, item count: {self.N_item}")

        S_item = np.zeros((self.N_item, self.N_item))
        np.fill_diagonal(S_item, 1)
        S_item_old = np.zeros((self.N_item, self.N_item))
        
        S_user = np.zeros((self.N_user, self.N_user))
        np.fill_diagonal(S_user, 1)
        S_user_old = np.zeros((self.N_user, self.N_user))
        
        G = (self.Rating > 0).astype(int)
        G_t = G.T

        start = time.time()
        Comb_user = pd.DataFrame(
                np.zeros((self.N_user, self.N_user)), 
                index = self.users, 
                columns = self.users)
        for i, row in G.iterrows():
            Comb_user.loc[i] = G.sum(axis = 1) * row.sum()
        Comb_user = Comb_user.values
        end = time.time()
        print(f"Comb_user matrix establised! {end - start}s spent.")
        
        start = time.time()
        Comb_item = pd.DataFrame(
                np.zeros((self.N_item, self.N_item)), 
                index = self.items, 
                columns = self.items)
        for i, row in G_t.iterrows():
            Comb_item.loc[i] = G_t.sum(axis = 1) * row.sum()
        Comb_item = Comb_item.values
        end = time.time()
        print(f"Comb_item matrix establised! {end - start}s spent.")

        for _iter in range(iterations):
            if self._converged(S_item, S_item_old, eps) and self._converged(S_user, S_user_old, eps):
                print(f"Converged at iteration {_iter}, break!")
                break
            
            print(f"Iteration {_iter + 1} / {iterations} start:")
            S_item_old = copy.deepcopy(S_item)
            S_user_old = copy.deepcopy(S_user)
            
            try:
                # Try with GPU
                print("Updating S_item with GPU...")
                start1 = time.time()
                S_item = C_item * run_dot_product(run_dot_product(G_t, S_user), G) / Comb_item
            except:
                # Go with CPU
                print("GPU failed, trying with CPU...")
                start1 = time.time()
                S_item = C_item * np.dot(np.dot(G_t, S_user), G) / Comb_item
            np.fill_diagonal(S_item, 1)
            end1 = time.time()

            try:
                # Try with GPU
                print("Updating S_user with GPU...")
                start2 = time.time()
                S_user = C_item * run_dot_product(run_dot_product(G, S_item), G_t) / Comb_user
            except:
                # Go with CPU
                start2 = time.time()
                print("GPU failed, trying with CPU...")
                S_user = C_user * np.dot(np.dot(G, S_item), G_t) / Comb_user
            np.fill_diagonal(S_user, 1)
            end2 = time.time()

            print(f"S_user updated in {end1 - start1}, S_item updated in {end2 - start2}!")
            
        self.S_item = pd.DataFrame(S_item, index = self.items, columns = self.items)
        self.S_user = pd.DataFrame(S_user, index = self.users, columns = self.users)

    def compare_train_test_set(self, test_data):
        unique_train_user = set(self.users)
        unique_train_item = set(self.items)
        unique_test_user = set(test_data.userId.unique())
        unique_test_item = set(test_data.movieId.unique())
        cold_start_user = unique_test_user - unique_test_user & unique_train_user
        cold_start_item = unique_test_item - unique_test_item & unique_train_item
        print(f"Count of cold start user: {len(cold_start_user)}")
        print(f"Count of cold start item: {len(cold_start_item)}")
        return cold_start_user, cold_start_item

    def cf_recommendation(self, df, cf_type = 'user', mask = True, melt = True):
        cold_start_user, cold_start_item = self.compare_train_test_set(df)
        user_to_predict = set(df.userId.unique()) - cold_start_user

        if cf_type == 'user':
            mean_user_rating = self.Rating.mean(axis=1)
            # Use np.newaxis so that mean_user_rating has same format as ratings
            ratings_diff = (self.Rating - mean_user_rating[:, np.newaxis])
            pred = mean_user_rating[:, np.newaxis] + self.S_user.dot(ratings_diff) / np.array([np.abs(self.S_user).sum(axis=1)]).T
        # elif cf_type == 'item':
        #     pred = ratings.dot(similarity) / np.array([np.abs(similarity).sum(axis=1)]) 
        pred = pd.DataFrame(pred, index = self.users, columns = self.items)
        pred = pred[pred.index.isin(user_to_predict)]
        if mask:
            pred = pred * (self.Rating == 0)

        if melt:
            pred = pd.melt(pred.reset_index(), id_vars = ['userId'], value_vars = pred.columns).dropna()
            pred = pred[pred['value'] != 0].rename(columns = {'value': 'rating'})
        return pred

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

    def fit(self, df, C_item = 0.8, C_user = 0.8, iterations = 100, eps = 1e-4):
        G_user, G_item = self._create_graph_from_df(df)
        self.users = G_user.index
        self.items = G_item.index
        self.N_user = len(self.users)
        self.N_item = len(self.items)
        print(f"User count: {self.N_user}, item count: {self.N_item}")

        print("Initializing User spread matrix...")
        start = time.time()
        spread_user = np.zeros((self.N_user, self.N_user))
        np.fill_diagonal(spread_user, G_user.var(axis = 1).apply(lambda x: np.exp(-x)))
        try:
            # Try with GPU
            W_user_item = run_dot_product(spread_user, G_user.fillna(0)) 
        except:
            # Go with CPU
            print("GPU failed, trying with CPU...")
            W_user_item = np.dot(spread_user, G_user.fillna(0))
        end = time.time()
        print(f"Finished in {end - start}s!")
        
        print("Initializing Item spread matrix...")
        start = time.time()
        spread_item = np.zeros((self.N_item, self.N_item))
        np.fill_diagonal(spread_item, G_item.var(axis = 1).apply(lambda x: np.exp(-x)))
        try:
            # Try with GPU
            W_item_user = run_dot_product(spread_user, G_item.fillna(0)) 
        except:
            # Go with CPU
            print("GPU failed, trying with CPU...")
            W_item_user = np.dot(spread_item, G_item.fillna(0))
        end = time.time()
        print(f"Finished in {end - start}s!")
        
        print("Initializing User evidence matrix...")
        start = time.time()
        try:
            # Try with GPU
            E_user = run_dot_product((G_user > 0).astype(int), (G_user > 0).T.astype(int)) 
        except:
            # Go with CPU
            print("GPU failed, trying with CPU...")
            E_user = np.dot((G_user > 0).astype(int), (G_user > 0).T.astype(int))
        E_user = 1 - 0.5**E_user
        end = time.time()
        print(f"Finished in {end - start}s!")
        
        print("Initializing Item evidence matrix...")
        start = time.time()
        try:
            # Try with GPU
            E_item = run_dot_product((G_item > 0).astype(int), (G_item > 0).T.astype(int)) 
        except:
            # Go with CPU
            print("GPU failed, trying with CPU...")
            E_item = np.dot((G_item > 0).astype(int), (G_item > 0).T.astype(int))
        E_item = 1 - 0.5**E_item
        end = time.time()
        print(f"Finished in {end - start}s!")

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
            
            try:
                # Try with GPU
                print("Updating S_item with GPU...")
                start1 = time.time()
                S_item = E_item * C_item * run_dot_product(run_dot_product(W_item_user, S_user), W_item_user.T)
            except:
                # Go with CPU
                print("GPU failed, trying with CPU...")
                start1 = time.time()
                S_item = E_item * C_item * np.dot(np.dot(W_item_user, S_user), W_item_user.T)
            np.fill_diagonal(S_item, 1)
            end1 = time.time()

            try:
                # Try with GPU
                print("Updating S_user with GPU...")
                start2 = time.time()
                S_user = E_item * C_item * run_dot_product(run_dot_product(W_user_item, S_item), W_user_item.T)
            except:
                # Go with CPU
                start2 = time.time()
                print("GPU failed, trying with CPU...")
                S_user = E_user * C_user * np.dot(np.dot(W_user_item, S_item), W_user_item.T)
            np.fill_diagonal(S_user, 1)
            end2 = time.time()

            print(f"S_user updated in {end1 - start1}, S_item updated in {end2 - start2}!")
            
        self.S_item = pd.DataFrame(S_item, index = self.items, columns = self.items)
        self.S_user = pd.DataFrame(S_user, index = self.users, columns = self.users)

    def compare_train_test_set(self, test_data):
        unique_train_user = set(self.users)
        unique_train_item = set(self.items)
        unique_test_user = set(test_data.userId.unique())
        unique_test_item = set(test_data.movieId.unique())
        cold_start_user = unique_test_user - unique_test_user & unique_train_user
        cold_start_item = unique_test_item - unique_test_item & unique_train_item
        print(f"Count of cold start user: {len(cold_start_user)}")
        print(f"Count of cold start item: {len(cold_start_item)}")
        return cold_start_user, cold_start_item

    def cf_recommendation(self, df, cf_type = 'user', mask = True, melt = True):
        cold_start_user, cold_start_item = self.compare_train_test_set(df)
        user_to_predict = set(df.userId.unique()) - cold_start_user

        if cf_type == 'user':
            mean_user_rating = self.G_rating.mean(axis=1)
            # Use np.newaxis so that mean_user_rating has same format as ratings
            ratings_diff = (self.G_rating - mean_user_rating[:, np.newaxis])
            pred = mean_user_rating[:, np.newaxis] + self.S_user.dot(ratings_diff) / np.array([np.abs(self.S_user).sum(axis=1)]).T
        # elif cf_type == 'item':
        #     pred = ratings.dot(similarity) / np.array([np.abs(similarity).sum(axis=1)]) 
        pred = pd.DataFrame(pred, index = self.users, columns = self.items)
        pred = pred[pred.index.isin(user_to_predict)]
        if mask:
            pred = pred * (self.G_rating == 0)

        if melt:
            pred = pd.melt(pred.reset_index(), id_vars = ['userId'], value_vars = pred.columns).dropna()
            pred = pred[pred['value'] != 0].rename(columns = {'value': 'rating'})
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

    def fit(self, df, tag, C_item = 0.8, C_user = 0.8, lbd = 0.3, iterations = 100, eps = 1e-4, how = 'cos'):
        G_user, G_item = self._create_graph_from_df(df)
        self.users = G_user.index
        self.items = G_item.index
        self.N_user = len(self.users)
        self.N_item = len(self.items)
        print(f"User count: {self.N_user}, item count: {self.N_item}")
        
        print("Initializing tab-based item similarity matrix...")
        start = time.time()
        tag = tag[tag.movieId.isin(self.items)]
        self.G_tag = tag.pivot(index = 'movieId', columns = 'tagId', values = 'relevance')
        if how == 'cos':
            self.S_tag_based = cosine_similarity(self.G_tag.fillna(0))
            
        elif how == 'jac':
            tag_or = pd.DataFrame(
                    np.zeros((self.N_item, self.N_item)),
                    index = self.items,
                    columns = self.items)
            for i, row in self.G_tag.iterrows():
                tag_or[i] = (self.G_tag + row).count(axis = 1)

            self.G_tag = self.G_tag.fillna(0)
            tag_and = np.dot(self.G_tag, self.G_tag.T)

            self.S_tag_based = (tag_and / tag_or).replace([np.inf, -np.inf], np.nan).fillna(0).values
        
        end = time.time()
        print(f"Finished in {end - start}s!")

        print("Initializing User spread matrix...")
        start = time.time()
        spread_user = np.zeros((self.N_user, self.N_user))
        np.fill_diagonal(spread_user, G_user.var(axis = 1).apply(lambda x: np.exp(-x)))
        try:
            # Try with GPU
            W_user_item = run_dot_product(spread_user, G_user.fillna(0)) 
        except:
            # Go with CPU
            print("GPU failed, trying with CPU...")
            W_user_item = np.dot(spread_user, G_user.fillna(0))
        end = time.time()
        print(f"Finished in {end - start}s!")
        
        print("Initializing Item spread matrix...")
        start = time.time()
        spread_item = np.zeros((self.N_item, self.N_item))
        np.fill_diagonal(spread_item, G_item.var(axis = 1).apply(lambda x: np.exp(-x)))
        try:
            # Try with GPU
            W_item_user = run_dot_product(spread_user, G_item.fillna(0)) 
        except:
            # Go with CPU
            print("GPU failed, trying with CPU...")
            W_item_user = np.dot(spread_item, G_item.fillna(0))
        end = time.time()
        print(f"Finished in {end - start}s!")
        
        print("Initializing User evidence matrix...")
        start = time.time()
        try:
            # Try with GPU
            E_user = run_dot_product((G_user > 0).astype(int), (G_user > 0).T.astype(int)) 
        except:
            # Go with CPU
            print("GPU failed, trying with CPU...")
            E_user = np.dot((G_user > 0).astype(int), (G_user > 0).T.astype(int))
        E_user = 1 - 0.5**E_user
        end = time.time()
        print(f"Finished in {end - start}s!")
        
        print("Initializing Item evidence matrix...")
        start = time.time()
        try:
            # Try with GPU
            E_item = run_dot_product((G_item > 0).astype(int), (G_item > 0).T.astype(int)) 
        except:
            # Go with CPU
            print("GPU failed, trying with CPU...")
            E_item = np.dot((G_item > 0).astype(int), (G_item > 0).T.astype(int))
        E_item = 1 - 0.5**E_item
        end = time.time()
        print(f"Finished in {end - start}s!")

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
            
            try:
                # Try with GPU
                print("Updating S_item with GPU...")
                start1 = time.time()
                S_item = E_item * C_item * run_dot_product(run_dot_product(W_item_user, S_user), W_item_user.T)
            except:
                # Go with CPU
                print("GPU failed, trying with CPU...")
                start1 = time.time()
                S_item = E_item * C_item * np.dot(np.dot(W_item_user, S_user), W_item_user.T)
            np.fill_diagonal(S_item, 1)
            
            S_item = (1 - lbd) * S_item + lbd * self.S_tag_based
            
            end1 = time.time()

            try:
                # Try with GPU
                print("Updating S_user with GPU...")
                start2 = time.time()
                S_user = E_item * C_item * run_dot_product(run_dot_product(W_user_item, S_item), W_user_item.T)
            except:
                # Go with CPU
                start2 = time.time()
                print("GPU failed, trying with CPU...")
                S_user = E_user * C_user * np.dot(np.dot(W_user_item, S_item), W_user_item.T)
            np.fill_diagonal(S_user, 1)
            end2 = time.time()

            print(f"S_user updated in {end1 - start1}, S_item updated in {end2 - start2}!")
            
        self.S_item = pd.DataFrame(S_item, index = self.items, columns = self.items)
        self.S_user = pd.DataFrame(S_user, index = self.users, columns = self.users)

    def compare_train_test_set(self, test_data):
        unique_train_user = set(self.users)
        unique_train_item = set(self.items)
        unique_test_user = set(test_data.userId.unique())
        unique_test_item = set(test_data.movieId.unique())
        cold_start_user = unique_test_user - unique_test_user & unique_train_user
        cold_start_item = unique_test_item - unique_test_item & unique_train_item
        print(f"Count of cold start user: {len(cold_start_user)}")
        print(f"Count of cold start item: {len(cold_start_item)}")
        return cold_start_user, cold_start_item

    def cf_recommendation(self, df, cf_type = 'user', mask = True, melt = True):
        cold_start_user, cold_start_item = self.compare_train_test_set(df)
        user_to_predict = set(df.userId.unique()) - cold_start_user

        if cf_type == 'user':
            mean_user_rating = self.G_rating.mean(axis=1)
            # Use np.newaxis so that mean_user_rating has same format as ratings
            ratings_diff = (self.G_rating - mean_user_rating[:, np.newaxis])
            pred = mean_user_rating[:, np.newaxis] + self.S_user.dot(ratings_diff) / np.array([np.abs(self.S_user).sum(axis=1)]).T
        # elif cf_type == 'item':
        #     pred = ratings.dot(similarity) / np.array([np.abs(similarity).sum(axis=1)]) 
        pred = pd.DataFrame(pred, index = self.users, columns = self.items)
        pred = pred[pred.index.isin(user_to_predict)]
        if mask:
            pred = pred * (self.G_rating == 0)

        if melt:
            pred = pd.melt(pred.reset_index(), id_vars = ['userId'], value_vars = pred.columns).dropna()
            pred = pred[pred['value'] != 0].rename(columns = {'value': 'rating'})
        return pred

