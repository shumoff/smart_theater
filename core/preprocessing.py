import time
import pandas as pd
import numpy as np


def strip_lists(lst):
    for i in range(len(lst)):
        lst[i] = lst[i].strip()
    return lst


def split_json_lists(data, label):
    objects_set = set()
    counter = 0
    print(data)
    for objects in data[label]:
        counter += 1
        objects_set.update(set(map(lambda x: x.strip(), objects)))
    objects_dict = dict(zip(sorted(objects_set), range(1, len(objects_set) + 1)))
    print(objects_dict)
    pd.DataFrame({'id': list(objects_dict.values()), f'{label}': list(objects_dict.keys())}).to_csv(
        path_or_buf=f'~/PycharmProjects/smart-theater/data/{label}.csv', index=False)
    return objects_dict


def get_conn_tables(data, label):
    counter = 0
    conn_table_df = pd.DataFrame(columns=['id', 'movie_id', f'{label}_id'])
    index_dict = split_json_lists(data, label)
    for index, row in data[['movieId', label]].iterrows():
        if counter % 1000 == 0:
            print(counter)
        for object_ in row[label]:
            conn_table_df.loc[len(conn_table_df)] = [index + 1, row['movieId'], index_dict[object_]]
        counter += 1
    conn_table_df.to_csv(path_or_buf=f'~/PycharmProjects/smart-theater/data/movie_{label}.csv', index=False)
    del conn_table_df


def preprocess_data():
    movies_df = pd.read_json(path_or_buf='~/PycharmProjects/smart-theater/data/movies.json')
    ratings_df = pd.read_csv('~/PycharmProjects/smart-theater/data/ratings.csv')

    movies_df = movies_df.dropna()
    movies_df['country'] = movies_df['country'].apply(lambda x: 0 if x[0] == 'N/A' else x)
    movies_df['director'] = movies_df['director'].apply(lambda x: 0 if x[0] == 'N/A' else x)
    movies_df['genre'] = movies_df['genre'].apply(lambda x: 0 if x[0] == 'N/A' else x)
    movies_df = movies_df[(movies_df['country'] != 0) & (movies_df['director'] != 0) & (movies_df['genre'] != 0)]

    movies_df['country'] = movies_df['country'].apply(strip_lists)
    movies_df['director'] = movies_df['director'].apply(strip_lists)
    movies_df['genre'] = movies_df['genre'].apply(strip_lists)
    movies_df = movies_df[movies_df['poster_link'] != 'N/A']

    movies_lists_df = movies_df[['movieId', 'country', 'genre', 'director']].copy()
    movies_df.drop(columns=['country', 'genre', 'director'], inplace=True)

    movies_df.drop_duplicates(subset=list(movies_df.columns)[1:], inplace=True)
    movie_ids = movies_df['movieId'].copy()
    movies_df.drop(columns=['movieId'], inplace=True)

    movies_lists_df = movies_lists_df[movies_lists_df['movieId'].isin(movie_ids)]
    ratings_df = ratings_df[ratings_df['movieId'].isin(movie_ids)]

    start_time = time.monotonic()
    ratings_df['movieId'] = ratings_df['movieId'].apply(lambda x: np.where(movie_ids == x)[0][0] + 1)
    print("Time of reindexing: ", time.monotonic() - start_time)

    start_time = time.monotonic()
    movies_lists_df['movieId'] = movies_lists_df['movieId'].apply(lambda x: np.where(movie_ids == x)[0][0] + 1)
    print("Time of reindexing: ", time.monotonic() - start_time)

    get_conn_tables(movies_lists_df, 'country')
    get_conn_tables(movies_lists_df, 'genre')
    get_conn_tables(movies_lists_df, 'director')

    start_time = time.monotonic()
    ratings_df['rating'] = ratings_df['rating'].apply(lambda x: int(x * 2))
    print("Time of recalculating: ", time.monotonic() - start_time)

    start_time = time.monotonic()
    ratings_df.rename(columns={'userId': 'user_id', 'movieId': 'movie_id'}, inplace=True)
    ratings_df.to_csv(path_or_buf='~/PycharmProjects/smart-theater/data/ratings_enhanced.csv', index=False)
    print("Time of writing to csv: ", time.monotonic() - start_time)
    del ratings_df

    movies_df.rename(columns={'imdbRating': 'imdb_rating'}, inplace=True)
    movies_df.to_csv(path_or_buf='~/PycharmProjects/smart-theater/data/movies_enhanced.csv', index=False)


def main():
    preprocess_data()


if __name__ == '__main__':
    main()
