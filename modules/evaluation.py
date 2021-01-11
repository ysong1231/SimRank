import pandas as pd
import numpy as np

COL_USER = "userId"
COL_ITEM = "movieId"
COL_RATING = "rating"
COL_PREDICTION = "rating"
DEFAULT_K = 10
DEFAULT_THRESHOLD = 12

DEFAULT_USER_COL = "userId"
DEFAULT_ITEM_COL = "movieId"
DEFAULT_RATING_COL = "rating"
#DEFAULT_LABEL_COL = "label"
#DEFAULT_TIMESTAMP_COL = "timestamp"
DEFAULT_PREDICTION_COL = "rating"

def group(df, to_group, thres= [5, 10, 50, 100, 500, 1000, 10000, 100000]):
    def get_group(x, thres=thres):
        for i in range(len(thres)):
            if x <= thres[i]:
                return i+1
            
    def get_group_range(x, thres=thres):
        for i in range(len(thres)):
            if x <= thres[i]:
                if i == 0:
                    return f'0-{thres[i]}'
                else:
                    return f'{thres[i-1]}-{thres[i]}'
            elif x > thres[-1]:
                return f'{thres[-1]}+'
    if to_group == 'user':
        df = df[['userId', 'movieId']].groupby('userId').count()
        df['groupId'] = df['movieId'].apply(lambda x: get_group(x))
        df['groupRange'] = df['movieId'].apply(lambda x: get_group_range(x)) 
        df = df.drop(['movieId'], axis = 1)
        df = df.reset_index()
        return df
    if to_group == 'movie':
        df = df[['movieId', 'userId']].groupby('movieId').count()
        df['groupId'] = df['userId'].apply(lambda x: get_group(x))
        df['groupRange'] = df['userId'].apply(lambda x: get_group_range(x)) 
        df = df.drop(['userId'], axis = 1)
        df = df.reset_index()
        return df

def merge_ranking_true_pred(
    rating_true,
    rating_pred,
    col_user,
    col_item,
    col_rating,
    col_prediction,
    relevancy_method,
    user_grouped,
    k=DEFAULT_K,
    threshold=DEFAULT_THRESHOLD,
    
):
    """Filter truth and prediction data frames on common users
    Args:
        rating_true (pd.DataFrame): True DataFrame
        rating_pred (pd.DataFrame): Predicted DataFrame
        col_user (str): column name for user
        col_item (str): column name for item
        col_rating (str): column name for rating
        col_prediction (str): column name for prediction
        relevancy_method (str): method for determining relevancy ['top_k', 'by_threshold', None]. None means that the 
            top k items are directly provided, so there is no need to compute the relevancy operation.
        k (int): number of top k items per user (optional)
        threshold (float): threshold of top items per user (optional)
    Returns:
        pd.DataFrame, pd.DataFrame, int: DataFrame of recommendation hits, sorted by `col_user` and `rank`
        DataFrmae of hit counts vs actual relevant items per user number of unique user ids
    """

    # Make sure the prediction and true data frames have the same set of users
    common_users = set(rating_true[col_user]).intersection(set(rating_pred[col_user]))
    rating_true_common = rating_true[rating_true[col_user].isin(common_users)]
    rating_pred_common = rating_pred[rating_pred[col_user].isin(common_users)]
    n_users = len(common_users)

    # Return hit items in prediction data frame with ranking information. This is used for calculating NDCG and MAP.
    # Use first to generate unique ranking values for each item. This is to align with the implementation in
    # Spark evaluation metrics, where index of each recommended items (the indices are unique to items) is used
    # to calculate penalized precision of the ordered items.        

    
    # Sort dataframe by col_user and (top k) col_rating
    if relevancy_method is None:
        df_hit = rating_pred_common
    elif relevancy_method == "top_k":
        df_hit = rating_pred_common.groupby(col_user, as_index=False)\
        .apply(lambda x: x.nlargest(k, col_prediction))\
        .reset_index(drop=True)
    elif relevancy_method == "by_threshold":
        df_hit = rating_pred_common[rating_pred_common[col_prediction] >= threshold]\
        .sort_values(col_prediction, ascending=False)
    else:
        raise NotImplementedError("Invalid relevancy_method")
        
    # Add ranks
    df_hit["rank"] = df_hit.groupby(col_user, sort=False).cumcount() + 1
    
    
    df_hit = pd.merge(df_hit, rating_true_common, on=[col_user, col_item])[
        [col_user, col_item, "rank"]
    ]

    # count the number of hits vs actual relevant items per user
    df_hit_count = pd.merge(
        df_hit.groupby(col_user, as_index=False)[col_user].agg({"hit": "count"}),
        rating_true_common.groupby(col_user, as_index=False)[col_user].agg(
            {"actual": "count"}
        ),
        on=col_user,
    )
    
    df_hit_count = df_hit_count.join(user_grouped.set_index(col_user), on = [col_user], how = 'left')

    return df_hit, df_hit_count, n_users

def precision_at_k(
    rating_true,
    rating_pred,
    col_user=DEFAULT_USER_COL,
    col_item=DEFAULT_ITEM_COL,
    col_rating=DEFAULT_RATING_COL,
    col_prediction=DEFAULT_PREDICTION_COL,
    relevancy_method="top_k",
    k=DEFAULT_K,
    user_grouped = None,
    threshold=DEFAULT_THRESHOLD,
):
    """Precision at K.
    Note:
        In particular, the maximum achievable precision may be < 1, if the number of items for a
        user in rating_pred is less than k.
    Args:
        rating_true (pd.DataFrame): True DataFrame
        rating_pred (pd.DataFrame): Predicted DataFrame
        col_user (str): column name for user
        col_item (str): column name for item
        col_rating (str): column name for rating
        col_prediction (str): column name for prediction
        relevancy_method (str): method for determining relevancy ['top_k', 'by_threshold', None]. None means that the 
            top k items are directly provided, so there is no need to compute the relevancy operation.
        k (int): number of top k items per user
        threshold (float): threshold of top items per user (optional)
    Returns:
        float: precision at k (min=0, max=1)
    """

    df_hit, df_hit_count, n_users = merge_ranking_true_pred(
        rating_true=rating_true,
        rating_pred=rating_pred,
        col_user=col_user,
        col_item=col_item,
        col_rating=col_rating,
        col_prediction=col_prediction,
        relevancy_method=relevancy_method,
        user_grouped = user_grouped,
        k=k,
        threshold=threshold,
    )

    if df_hit.shape[0] == 0:
        return 0.0

    return (df_hit_count["hit"] / k).sum() / n_users

def recall_at_k(
    rating_true,
    rating_pred,
    col_user=DEFAULT_USER_COL,
    col_item=DEFAULT_ITEM_COL,
    col_rating=DEFAULT_RATING_COL,
    col_prediction=DEFAULT_PREDICTION_COL,
    relevancy_method="top_k",
    k=DEFAULT_K,
    user_grouped = None,
    threshold=DEFAULT_THRESHOLD,
    grouped_recall_return = False
):
    """Recall at K.
    Args:
        rating_true (pd.DataFrame): True DataFrame
        rating_pred (pd.DataFrame): Predicted DataFrame
        col_user (str): column name for user
        col_item (str): column name for item
        col_rating (str): column name for rating
        col_prediction (str): column name for prediction
        relevancy_method (str): method for determining relevancy ['top_k', 'by_threshold', None]. None means that the 
            top k items are directly provided, so there is no need to compute the relevancy operation.
        k (int): number of top k items per user
        threshold (float): threshold of top items per user (optional)
    Returns:
        float: recall at k (min=0, max=1). The maximum value is 1 even when fewer than 
        k items exist for a user in rating_true.
    """

    df_hit, df_hit_count, n_users = merge_ranking_true_pred(
        rating_true=rating_true,
        rating_pred=rating_pred,
        col_user=col_user,
        col_item=col_item,
        col_rating=col_rating,
        col_prediction=col_prediction,
        relevancy_method=relevancy_method,
        k=k,
        user_grouped=user_grouped,
        threshold=threshold,
    )
    
    if df_hit.shape[0] == 0:
        return 0.0
    
    df_hit_count['recall'] = df_hit_count["hit"] / df_hit_count["actual"]
    avg_recall = df_hit_count['recall'].sum() / n_users
    
    if grouped_recall_return:
        grouped_recall = df_hit_count.groupby(['groupId', 'groupRange']).agg({'recall': 'mean'}).reset_index()
        return grouped_recall, avg_recall

    return avg_recall

def ndcg_at_k(
    rating_true,
    rating_pred,
    col_user=DEFAULT_USER_COL,
    col_item=DEFAULT_ITEM_COL,
    col_rating=DEFAULT_RATING_COL,
    col_prediction=DEFAULT_PREDICTION_COL,
    relevancy_method="top_k",
    k=DEFAULT_K,
    user_grouped=None,
    threshold=DEFAULT_THRESHOLD,
):
    """Normalized Discounted Cumulative Gain (nDCG).
    
    Info: https://en.wikipedia.org/wiki/Discounted_cumulative_gain
    
    Args:
        rating_true (pd.DataFrame): True DataFrame
        rating_pred (pd.DataFrame): Predicted DataFrame
        col_user (str): column name for user
        col_item (str): column name for item
        col_rating (str): column name for rating
        col_prediction (str): column name for prediction
        relevancy_method (str): method for determining relevancy ['top_k', 'by_threshold', None]. None means that the 
            top k items are directly provided, so there is no need to compute the relevancy operation.
        k (int): number of top k items per user
        threshold (float): threshold of top items per user (optional)
    Returns:
        float: nDCG at k (min=0, max=1).
    """

    df_hit, df_hit_count, n_users = merge_ranking_true_pred(
        rating_true=rating_true,
        rating_pred=rating_pred,
        col_user=col_user,
        col_item=col_item,
        col_rating=col_rating,
        col_prediction=col_prediction,
        relevancy_method=relevancy_method,
        user_grouped=user_grouped,
        k=k,
        threshold=threshold,
    )

    if df_hit.shape[0] == 0:
        return 0.0

    # calculate discounted gain for hit items
    df_dcg = df_hit.copy()
    # relevance in this case is always 1
    df_dcg["dcg"] = 1 / np.log1p(df_dcg["rank"])
    # sum up discount gained to get discount cumulative gain
    df_dcg = df_dcg.groupby(col_user, as_index=False, sort=False).agg({"dcg": "sum"})
    # calculate ideal discounted cumulative gain
    df_ndcg = pd.merge(df_dcg, df_hit_count, on=[col_user])
    df_ndcg["idcg"] = df_ndcg["actual"].apply(
        lambda x: sum(1 / np.log1p(range(1, min(x, k) + 1)))
    )

    # DCG over IDCG is the normalized DCG
    return (df_ndcg["dcg"] / df_ndcg["idcg"]).sum() / n_users

def map_at_k(
    rating_true,
    rating_pred,
    col_user=DEFAULT_USER_COL,
    col_item=DEFAULT_ITEM_COL,
    col_rating=DEFAULT_RATING_COL,
    col_prediction=DEFAULT_PREDICTION_COL,
    relevancy_method="top_k",
    k=DEFAULT_K,
    user_grouped=None,
    threshold=DEFAULT_THRESHOLD,
):
    """Mean Average Precision at k
    
    The implementation of MAP is referenced from Spark MLlib evaluation metrics.
    https://spark.apache.org/docs/2.3.0/mllib-evaluation-metrics.html#ranking-systems
    A good reference can be found at:
    http://web.stanford.edu/class/cs276/handouts/EvaluationNew-handout-6-per.pdf
    Note:
        1. The evaluation function is named as 'MAP is at k' because the evaluation class takes top k items for
        the prediction items. The naming is different from Spark.
        
        2. The MAP is meant to calculate Avg. Precision for the relevant items, so it is normalized by the number of
        relevant items in the ground truth data, instead of k.
    Args:
        rating_true (pd.DataFrame): True DataFrame
        rating_pred (pd.DataFrame): Predicted DataFrame
        col_user (str): column name for user
        col_item (str): column name for item
        col_rating (str): column name for rating
        col_prediction (str): column name for prediction
        relevancy_method (str): method for determining relevancy ['top_k', 'by_threshold', None]. None means that the 
            top k items are directly provided, so there is no need to compute the relevancy operation.
        k (int): number of top k items per user
        threshold (float): threshold of top items per user (optional)
    Returns:
        float: MAP at k (min=0, max=1).
    """

    df_hit, df_hit_count, n_users = merge_ranking_true_pred(
        rating_true=rating_true,
        rating_pred=rating_pred,
        col_user=col_user,
        col_item=col_item,
        col_rating=col_rating,
        col_prediction=col_prediction,
        relevancy_method=relevancy_method,
        user_grouped=user_grouped,
        k=k,
        threshold=threshold,
    )

    if df_hit.shape[0] == 0:
        return 0.0

    # calculate reciprocal rank of items for each user and sum them up
    df_hit_sorted = df_hit.copy()
    df_hit_sorted["rr"] = (
        df_hit_sorted.groupby(col_user).cumcount() + 1
    ) / df_hit_sorted["rank"]
    df_hit_sorted = df_hit_sorted.groupby(col_user).agg({"rr": "sum"}).reset_index()

    df_merge = pd.merge(df_hit_sorted, df_hit_count, on=col_user)
    
    return (df_merge["rr"] / df_merge["actual"]).sum() / n_users

