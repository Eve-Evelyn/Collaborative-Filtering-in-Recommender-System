import pandas as pd
from sklearn.metrics.pairwise import cosine_similarity
import numpy as np

# read the csv file
data = pd.read_csv("Amazon.csv")
# set user_id column as dataframe index
data = data.set_index('user_id')

# check if there is any user_id duplicate data
# print(data['user_id'].duplicated().any())


def baseline_prediction(data, userid, movieid):
    """Function to calculate baseline prediction from user and movie """

    # calculate global mean
    global_mean = data.stack().dropna().mean()

    # calculate user mean
    user_mean = data.loc[userid, :].mean()

    # calculate item mean
    item_mean = data.loc[:, movieid].mean()

    # calculate user bias
    user_bias = global_mean - user_mean

    # calculate item bias
    item_bias = global_mean - item_mean

    # calculate baseline
    baseline_ui = global_mean + user_bias + item_bias

    return baseline_ui


# calculate the mean rating from all user for each movie
user_mean = data.mean(axis=0)
user_removed_mean_rating = (data - user_mean).fillna(0)


def find_neighbor(user_removed_mean_rating, userid, k=5):
    # Generate the similarity score
    n_users = len(user_removed_mean_rating.index)
    similarity_score = np.zeros(n_users)

    # get user 1 rating vector
    user_target = user_removed_mean_rating.loc[userid].values.reshape(1, -1)

    # Iterate all users
    for i, neighbor in enumerate(user_removed_mean_rating.index):
        # Extract neighbor user vector
        user_neighbor = user_removed_mean_rating.loc[neighbor].values.reshape(1, -1)

        # Calculate the similarity (we use Cosine Similarity)
        sim_i = cosine_similarity(user_target, user_neighbor)

        # Append
        similarity_score[i] = sim_i

    # Sort in descending orders of similarity_score
    sorted_idx = np.argsort(similarity_score)[::-1]

    # sort similarity score , descending
    similarity_score = np.sort(similarity_score)[::-1]

    # get user closest neighbor
    closest_neighbor = user_removed_mean_rating.index[sorted_idx[1:k + 1]].tolist()

    # slice neighbour similarity
    neighbor_similarity = list(similarity_score[1:k + 1])

    # return closest_neighbor
    return {
        'closest_neighbor': closest_neighbor,
        'closest_neighbor_similarity': neighbor_similarity
    }


def predict_item_rating(userid, movieid, data, neighbor_data, k,
                        max_rating=5, min_rating=1):
    """Function to predict rating on userid and movieid"""

    # calculate baseline (u,i)
    baseline = baseline_prediction(data=data,
                                   userid=userid, movieid=movieid)
    # for sum
    sim_rating_total = 0
    similarity_sum = 0
    # loop all over neighbor
    for i in range(k):
        # retrieve rating from neighbor
        neighbour_rating = data.loc[neighbor_data['closest_neighbor'][i], movieid]

        # skip if nan
        if np.isnan(neighbour_rating):
            continue

        # calculate baseline (ji)
        baseline = baseline_prediction(data=data,
                                       userid=neighbor_data['closest_neighbor'][i], movieid=movieid)

        # substract baseline from rating
        adjusted_rating = neighbour_rating - baseline

        # multiply by similarity
        sim_rating = neighbor_data['closest_neighbor_similarity'][i] * adjusted_rating

        # sum similarity * rating
        sim_rating_total += sim_rating

        #
        similarity_sum += neighbor_data['closest_neighbor_similarity'][i]

    # avoiding ZeroDivisionError
    try:
        user_item_predicted_rating = baseline + (sim_rating_total / similarity_sum)

    except ZeroDivisionError:
        user_item_predicted_rating = baseline

    # checking the boundaries of rating,
    if user_item_predicted_rating > max_rating:
        user_item_predicted_rating = max_rating

    elif user_item_predicted_rating < min_rating:
        user_item_predicted_rating = min_rating

    return user_item_predicted_rating


def recommend_items(data, userid, n_neighbor, n_items,
                    recommend_seen=False):
    """ Function to generate recommendation on given user_id """

    # find neighbor
    neighbor_data = find_neighbor(user_removed_mean_rating=user_removed_mean_rating,
                                  userid=userid, k=n_neighbor)

    # create empty dataframe to store prediction result
    prediction_df = pd.DataFrame()
    # create list to store prediction result
    predicted_ratings = []

    # mask seen item
    mask = np.isnan(data.loc[userid])
    item_to_predict = data.columns[mask]

    if recommend_seen:
        item_to_predict = data.columns

    # loop all over movie
    for movie in item_to_predict:
        # predict rating
        preds = predict_item_rating(userid=userid, movieid=movie,
                                    data=data,
                                    neighbor_data=neighbor_data, k=5)

        # append
        predicted_ratings.append(preds)

    # assign movieId
    prediction_df['movieId'] = data.columns[mask]

    # assign prediction result
    prediction_df['predicted_ratings'] = predicted_ratings

    #
    prediction_df = (prediction_df
                     .sort_values('predicted_ratings', ascending=False)
                     .head(n_items))

    return prediction_df


user_1_recommendation = recommend_items(data=data, userid='A3R5OBKS7OM2IR', n_neighbor=5, n_items=5,
                                        recommend_seen=False)
print(user_1_recommendation)
