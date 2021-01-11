import numpy as np
import pandas as pd

def align_items(ratings, tags):
    unique_movies_in_rating = ratings.movieId.unique()
    unique_movies_in_tag = tags.movieId.unique()
    ratings = ratings[ratings.movieId.isin(unique_movies_in_tag)]
    tags = tags[tags.movieId.isin(unique_movies_in_rating)]
    return ratings, tags

def sample_df(df, user_thresh=20, item_thresh=500, user_sample_n=20000, item_sample_n = 1000, random_seed=0):
    np.random.seed(random_seed)
    countItem = df[['movieId', 'rating']].groupby(['movieId']).count()
    selectedItemId = countItem.loc[countItem['rating'] > item_thresh].index
    #selectedItemId = random.sample(list(selectedItemId), item_sample_n)
    selectedItemId = np.random.choice(selectedItemId, item_sample_n, replace=False)
    df_sample_item = df[df['movieId'].isin(selectedItemId)]


    countUser = df_sample_item[['userId', 'rating']].groupby(['userId']).count()
    selectedUserId = countUser.loc[countUser['rating'] > user_thresh].index
    #selectedUserId = random.sample(list(selectedUserId), user_sample_n)
    selectedUserId = np.random.choice(selectedUserId, user_sample_n, replace=False)
    df_sample = df_sample_item[df_sample_item['userId'].isin(selectedUserId)]    
    
    n_users = len(df_sample.userId.unique())
    n_items = len(df_sample.movieId.unique())
    n_ratings = len(df_sample) 
    print(f'number of users: {n_users}')
    print(f'number of items: {n_items}')
    print(f'number of ratings: {n_ratings}')
    return df_sample