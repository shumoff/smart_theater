import numpy as np
import pandas as pd
from sklearn.metrics.pairwise import pairwise_distances
from sklearn.feature_extraction.text import TfidfVectorizer, CountVectorizer


def naive_predict(ratings_matrix, sim_amount, item_based=False):
    if item_based:
        ratings_matrix = ratings_matrix.T

    n_x = ratings_matrix.shape[0]
    n_y = ratings_matrix.shape[1]
    predictions = np.zeros((n_x, n_y))
    similarity_matrix = pairwise_distances(ratings_matrix, metric='cosine')

    for i in range(n_x):
        similar_items = similarity_matrix[i].argsort()[1:sim_amount + 1]
        sim_ratings_matrix = ratings_matrix[similar_items]
        predictions[i] = sim_ratings_matrix.sum(axis=0) / np.count_nonzero(sim_ratings_matrix, axis=0)

    if item_based:
        predictions = predictions.T

    return predictions


def weight_predict(ratings_matrix, sim_amount, item_based=False):
    if item_based:
        ratings_matrix = ratings_matrix.T

    n_x = ratings_matrix.shape[0]
    n_y = ratings_matrix.shape[1]
    predictions = np.zeros((n_x, n_y))
    similarity_matrix = pairwise_distances(ratings_matrix, metric='cosine')

    for i in range(n_x):
        user_similarity = similarity_matrix[i]
        similar_indexes = user_similarity.argsort()[1:sim_amount + 1].astype(np.int)
        weights = np.array([1 - x for x in user_similarity[similar_indexes]])
        sim_ratings_matrix = ratings_matrix[similar_indexes]
        numerators = weights.dot(sim_ratings_matrix)
        denominators = weights.dot(sim_ratings_matrix > 0)
        predictions[i] = numerators / denominators

    if item_based:
        predictions = predictions.T

    return predictions


def weight_mean_predict(ratings_matrix, sim_amount, item_based=False):
    if item_based:
        ratings_matrix = ratings_matrix.T

    n_x = ratings_matrix.shape[0]
    n_y = ratings_matrix.shape[1]
    predictions = np.zeros((n_x, n_y))
    similarity_matrix = pairwise_distances(ratings_matrix, metric='cosine')

    for i in range(n_x):
        user_similarity = similarity_matrix[i]
        similar_indexes = user_similarity.argsort()[1:sim_amount + 1].astype(np.int)
        weights = np.array([1 - x for x in user_similarity[similar_indexes]])
        sim_ratings_matrix = ratings_matrix[similar_indexes]
        mean_rating = ratings_matrix[i][ratings_matrix[i] > 0].mean()
        mean_rating = 0 if np.isnan(mean_rating) else mean_rating
        sim_users_mean_vector = np.ma.array(sim_ratings_matrix, mask=sim_ratings_matrix == 0).mean(axis=1)
        mean_deviation_matrix = np.where(sim_ratings_matrix == 0, 0, (sim_ratings_matrix.T - sim_users_mean_vector).T)
        numerators = weights.dot(mean_deviation_matrix)
        denominators = weights.dot(sim_ratings_matrix > 0)
        predictions[i] = mean_rating + np.nan_to_num(numerators / denominators)
        indexes = []
        for index, rating in enumerate(predictions[i]):
            if rating < 0:
                indexes.append(index)
        for index in indexes:
            predictions[i][index] = 0

    if item_based:
        predictions = predictions.T

    return predictions


def predict_user(user_id, n_users, n_movies, data):  # TODO: отдельной функцией сделать гибридные рекомендации
    sim_amount = 100
    ratings_matrix = np.zeros((n_users, n_movies))
    data_rows = list(map(lambda x: x.user_id - 1, data))
    data_cols = list(map(lambda x: x.movie_id - 1, data))
    data_vals = list(map(lambda x: x.rating, data))
    del data
    ratings_matrix[data_rows, data_cols] = data_vals
    user_ratings = ratings_matrix[user_id - 1]

    # calculate and write user's similarity
    similarity_vector = pairwise_distances(user_ratings.reshape(1, -1), ratings_matrix, metric='cosine')[0]

    similarity_df = pd.Dataframe(columns=['user_id', 'movie_id', 'rating'])
    similarity_df['i_user_id'] = [user_id] * n_users
    similarity_df['j_user_id'] = np.arange(1, n_users + 1)
    similarity_df['sim_coef'] = similarity_vector

    # predict user's ratings and write them
    similar_indexes = similarity_vector.argsort()[1:sim_amount + 1]
    weights = np.array([1 - x for x in similarity_vector[similar_indexes]])
    sim_ratings_matrix = ratings_matrix[similar_indexes]
    mean_rating = ratings_matrix[user_id - 1][ratings_matrix[user_id - 1] > 0].mean()
    mean_rating = 0 if np.isnan(mean_rating) else mean_rating
    sim_users_mean_vector = np.ma.array(sim_ratings_matrix, mask=sim_ratings_matrix == 0).mean(axis=1)
    mean_deviation_matrix = np.where(sim_ratings_matrix == 0, 0, (sim_ratings_matrix.T - sim_users_mean_vector).T)
    numerators = weights.dot(mean_deviation_matrix)
    denominators = weights.dot(sim_ratings_matrix > 0)
    predictions = mean_rating + np.nan_to_num(numerators / denominators)

    predictions = map(lambda x: 0 if x < 0 else x, predictions)

    predictions_df = pd.DataFrame(columns=['user_id', 'movie_id', 'rating'])
    predictions_df['user'] = [user_id] * n_movies
    predictions_df['movie_id'] = np.arange(1, n_movies + 1)
    predictions_df['rating'] = predictions

    return predictions_df, similarity_df


def similar_movies(movie_id, n_users, n_movies, data):
    ratings_matrix = np.zeros((n_movies, n_users))
    data_rows = list(map(lambda x: x.movie_id - 1, data))
    data_cols = list(map(lambda x: x.user_id - 1, data))
    data_vals = list(map(lambda x: x.rating, data))
    del data
    ratings_matrix[data_rows, data_cols] = data_vals
    movie_ratings = ratings_matrix[movie_id - 1]

    # calculate and write movie's similarity
    similarity_vector = 1 - pairwise_distances(movie_ratings.reshape(1, -1), ratings_matrix, metric='cosine')[0]

    similarity_df = pd.DataFrame(columns=['movie_id', 'other_movie_id', 'sim_coef'])
    similarity_df['movie_id'] = [movie_id] * n_movies
    similarity_df['other_movie_id'] = np.arange(1, n_movies + 1)
    similarity_df['sim_coef'] = similarity_vector

    return similarity_df


def search_movie(query, titles, descriptions, offset):  # TODO: придумать не узколобый подход к поиску по описанию
    sim_amount = 10
    vectorizer = TfidfVectorizer()
    titles.append(query)
    descriptions.append(query)
    term_title_matrix = vectorizer.fit_transform(titles).toarray()
    # term_description_matrix = vectorizer.fit_transform(descriptions).toarray()
    titles_similarity_vector = pairwise_distances(
        term_title_matrix[-1].reshape(1, -1), term_title_matrix, metric='cosine')[0]
    # descriptions_similarity_vector = pairwise_distances(
    #     term_description_matrix[-1].reshape(1, -1), term_description_matrix, metric='cosine')[0]
    similarity_vector = titles_similarity_vector  # + descriptions_similarity_vector
    similar_indexes = similarity_vector[:-1].argsort()[:sim_amount]
    return dict(zip(similar_indexes + offset + 1, 1 - similarity_vector[similar_indexes]))

