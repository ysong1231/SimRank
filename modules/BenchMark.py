import pandas as pd
import numpy as np
from sklearn.metrics.pairwise import cosine_similarity

class cf_recommendation:
    def __init__(self):
        self.Rating = None
        self.users = None
        self.items = None

    def _create_graph_from_df(self, df):
        self.Rating = df.pivot(index = 'userId', columns = 'movieId', values = 'rating').fillna(0)
        self.users = self.Rating.index
        self.items = self.Rating.columns

    def fit(self, df, how = 'cos'):
        self._create_graph_from_df(df)
        if how == 'cos':
            S = cosine_similarity(self.Rating)
            self.S = pd.DataFrame(S, index = self.users, columns = self.users)
        elif how == 'pearson':
            self.S = self.Rating.corr(method = 'pearson').fillna(0)
        else:
            raise NotImplementedError(f"Method {how} is not implemented...")
    
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

    def predict(self, df, k = 10, mask = True, melt = True):
        cold_start_user, cold_start_item = self.compare_train_test_set(df)
        user_to_predict = set(df.userId.unique()) - cold_start_user

        R = self.Rating.replace(0, np.nan)
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
            k_neighbors = set(self.S.loc[user].nlargest(k + 1).index) - set([user])
            s_neighbors = self.S[self.S.index.isin(k_neighbors)][user]
            s_neighbors = s_neighbors / s_neighbors.sum()
            diff_neighbors = diff[diff.index.isin(k_neighbors)]
            diag = np.zeros((k, k))
            np.fill_diagonal(diag, s_neighbors)
            weighted_diff = diag.dot(diff_neighbors).sum(axis = 0)
            pred.loc[user] = weighted_diff + mean_user_rating[user]
            if mask:
                pred.loc[user] = pred.loc[user] * (self.Rating.loc[user] == 0)
            curr += 1
            if curr / total >= poc:
                print(f'{curr}/{total} completed')
                poc += 0.1
        if melt:
            pred = pd.melt(pred.reset_index(), id_vars = ['userId'], value_vars = pred.columns).dropna()
            pred = pred[pred['value'] != 0].rename(columns = {'value': 'rating', 'variable': 'movieId'})
        return pred