import config
import pandas as pd
import numpy as np
import time
import hashlib
import random
from sqlalchemy import create_engine
from sqlalchemy.ext.declarative import declarative_base
from sqlalchemy import Column, Integer, String, ForeignKey, Float, Boolean, Text, desc
from sqlalchemy.orm import sessionmaker
from sqlalchemy import or_


dbname = "smart_theater"
user_name = "shumoff"
db_password = "$>g+3TcR5&k&3[X3"
engine = create_engine(f"mysql+pymysql://{user_name}:{db_password}@127.0.0.1/{dbname}", encoding='utf-8', echo=False)
"mysql://scott:tiger@hostname/dbname"
session = sessionmaker(bind=engine)
base = declarative_base()


class User(base):
    __tablename__ = 'users'

    id = Column(Integer, primary_key=True)
    token_id = Column(ForeignKey('tokens.id'))
    username = Column(String(64))
    email = Column(String(64))
    password = Column(String(64))
    timestamp_joined = Column(Integer)
    offset_similar = Column(Integer)
    offset_personal = Column(Integer)
    offset_hybrid = Column(Integer)

    def __repr__(self):
        return f"<User(nickname={self.username}, mail={self.email}, password={self.password}, " \
               f"token_id={self.token_id}," f"timestamp_joined={self.timestamp_joined}, " \
               f"offset_similar={self.offset_similar}, " f"offset_personal={self.offset_personal}, " \
               f"offset_hybrid={self.offset_hybrid})>"


class Token(base):
    __tablename__ = 'tokens'

    id = Column(Integer, primary_key=True)
    token = Column(String(256))
    created_timestamp = Column(Integer)
    expiration_timestamp = Column(Integer)

    def __repr__(self):
        return f"<Token(token={self.token})>"


class Movie(base):
    __tablename__ = 'movies'

    id = Column(Integer, primary_key=True)
    title = Column(String(256))
    year = Column(Integer)
    runtime = Column(Integer)
    imdb_rating = Column(Float)
    description = Column(Text)
    poster_link = Column(String(256))
    link = Column(String(256))

    def __repr__(self):
        return f"<Movie(title={self.title}, year={self.year}, imdb_rating={self.imdb_rating}, runtime={self.runtime}," \
            f" description={self.description}, poster_link={self.poster_link}, link={self.link})>"


class Rating(base):
    __tablename__ = 'ratings'

    id = Column(Integer, primary_key=True)
    user_id = Column(ForeignKey('users.id'))
    movie_id = Column(ForeignKey('movies.id'))
    rating = Column(Integer)
    timestamp = Column(Integer)

    def __repr__(self):
        return f"<Rating(user_id={self.user_id}, movie_id={self.movie_id}, rating={self.rating}, " \
            f"timestamp_watched={self.timestamp})>"


class Similarity(base):
    __tablename__ = 'similarity'

    id = Column(Integer, primary_key=True)
    i_user_id = Column(ForeignKey('users.id'))
    j_user_id = Column(ForeignKey('users.id'))
    sim_coef = Column(Float)

    def __repr__(self):
        return f"<Similarity(i_user_id={self.i_user_id}, j_user_id={self.j_user_id}, sim_coef={self.sim_coef})>"


class Prediction(base):
    __tablename__ = 'predictions'

    id = Column(Integer, primary_key=True)
    user_id = Column(ForeignKey('users.id'))
    movie_id = Column(ForeignKey('movies.id'))
    rating = Column(Float)

    def __repr__(self):
        return f"<Rating(user_id={self.user_id}, movie_id={self.movie_id}, rating={self.rating})>"


class NewUserState(base):
    __tablename__ = 'new_users_state'

    id = Column(Integer, primary_key=True)
    user_id = Column(ForeignKey('users.id'))
    sent_movie_id = Column(ForeignKey('movies.id'))

    def __repr__(self):
        return f"<Rating(user_id={self.user_id}, sent_movie_id={self.sent_movie_id})>"


class Country(base):
    __tablename__ = 'countries'

    id = Column(Integer, primary_key=True)
    name = Column(String(256))

    def __repr__(self):
        return f"<Country(name={self.name})>"


class Genre(base):
    __tablename__ = 'genres'

    id = Column(Integer, primary_key=True)
    name = Column(String(256))

    def __repr__(self):
        return f"<Genre(name={self.name})>"


class Director(base):
    __tablename__ = 'directors'

    id = Column(Integer, primary_key=True)
    name = Column(String(256))

    def __repr__(self):
        return f"<Director(name={self.name})>"


class MovieCountry(base):
    __tablename__ = 'movie_country'

    id = Column(Integer, primary_key=True)
    movie_id = Column(ForeignKey('movies.id'))
    country_id = Column(ForeignKey('countries.id'))

    def __repr__(self):
        return f"<MovieCountry(movie_id={self.movie_id}, country_id={self.country_id})>"


class MovieGenre(base):
    __tablename__ = 'movie_genre'

    id = Column(Integer, primary_key=True)
    movie_id = Column(ForeignKey('movies.id'))
    genre_id = Column(ForeignKey('genres.id'))

    def __repr__(self):
        return f"<MovieGenre(movie_id={self.movie_id}, genre_id={self.genre_id})>"


class MovieDirector(base):
    __tablename__ = 'movie_director'

    id = Column(Integer, primary_key=True)
    movie_id = Column(ForeignKey('movies.id'))
    director_id = Column(ForeignKey('directors.id'))

    def __repr__(self):
        return f"<MovieDirector(movie_id={self.movie_id}, director_id={self.director_id})>"


def random_hash(username):
    sha_methods = dict(
        sha1=hashlib.sha1,
        sha224=hashlib.sha224,
        sha256=hashlib.sha256,
        sha384=hashlib.sha384,
        sha512=hashlib.sha512,
    )
    token = sha_methods[random.choice(list(sha_methods.keys()))](username.encode('utf-8')).hexdigest()
    return token


def insert_vector(vector, table, user_id, similarity=False):

    n_x = len(vector)
    print(vector[:5])

    if similarity:
        df = pd.DataFrame(data=np.array([user_id] * n_x).reshape(n_x, 1), columns=["i_user_id"])
        df["j_user_id"] = np.array([i + 1 for i in range(n_x)]).reshape(n_x, 1)
        df["sim_coef"] = vector
        del vector

    else:
        df = pd.DataFrame(data=np.array([user_id] * n_x).reshape((n_x, 1)), columns=["user_id"])
        df["movie_id"] = np.array([i + 1 for i in range(n_x)]).reshape(n_x, 1)
        df["rating"] = vector
        del vector

    print(df.head(10))
    df.to_sql(con=engine, index=False, name=table.__tablename__, if_exists='append', chunksize=200000)


def add_new_user(username, email, password):
    sess = session()
    username_found = sess.query(User).filter(User.username == username).first()
    email_found = sess.query(User).filter(User.email == email).first()
    sess.close()
    if username_found:
        return {'OK': False, 'error': 'username_taken'}
    elif email_found:
        return {'OK': False, 'error': 'mail_taken'}
    else:
        token = random_hash(username)
        add_new_token(token)
        sess = session()
        token_id = sess.query(Token.id).order_by(Token.id.desc()).first()[0]
        sess.close()
        user_data = {'token_id': [token_id], 'username': [username], 'email': [email], 'password': [password],
                     'timestamp_joined': [time.time()], 'offset_similar': [0], 'offset_personal': [0],
                     'offset_hybrid': [0]}
        users_df = pd.DataFrame(data=user_data)
        users_df.to_sql(con=engine, index=False, name=User.__tablename__, if_exists='append')
        return {'OK': True, 'token': token}


def check_logpass(login, password):
    sess = session()
    res = sess.query(User.password, User.username).filter(or_(User.username == login, User.email == login)).first()
    sess.close()
    if res:
        db_pass, username = res
        if db_pass == password:
            return {'OK': True, 'username': username}
        else:
            return {'OK': False, 'error': 'pass_incorrect'}
    else:
        return {'OK': False, 'error': 'login_incorrect'}


def add_new_token(token):
    token_data = {'token': [token], 'created_timestamp': [time.time()],
                  'expiration_timestamp': [time.time() + config.TOKEN_LIFETIME]}
    tokens_df = pd.DataFrame(data=token_data)
    tokens_df.to_sql(con=engine, index=False, name=Token.__tablename__, if_exists='append')


def renew_token(username):
    sess = session()
    token_id = sess.query(User.token_id).filter(User.username == username).first()[0]
    db_token = sess.query(Token).filter(Token.id == token_id).first()
    token = random_hash(username)
    db_token.token = token
    db_token.created_timestamp = time.time()
    db_token.expiration_timestamp = time.time() + config.TOKEN_LIFETIME
    sess.commit()
    sess.close()
    return token


def spoil_token(token):
    sess = session()
    resp = sess.query(Token).filter(Token.token == token).first()
    if resp:  # если две машины, то на старой может быть неактуальный токен
        if resp.expiration_timestamp > resp.created_timestamp:
            resp.expiration_timestamp -= config.TOKEN_LIFETIME
            sess.commit()
    sess.close()


def check_token(token):
    sess = session()
    resp = sess.query(Token).filter(Token.token == token).first()
    if resp:
        token_id = resp.token_id
        expiration_timestamp = resp.expiration_timestamp
        username = sess.query(User).filter(User.token_id == token_id).first().username
        sess.close()
        if time.time() < expiration_timestamp:
            return {'OK': True, 'username': username}
        else:
            return {'OK': False}
    else:
        sess.close()
        return {'OK': False}


def get_ratings_matrix():
    sess = session()
    users_amount = sess.query(User).count()
    movies_amount = sess.query(Movie).count()
    resp = sess.query(Rating).all()
    sess.close()
    return {'users_amount': users_amount, 'movies_amount': movies_amount, 'data': resp}


def user_status(username):
    sess = session()
    user_id = sess.query(User).filter(User.username == username).first().id
    watched_movies = sess.query(Rating).filter(Rating.user_id == user_id).count()
    sess.close()
    return {'status': watched_movies >= config.MIN_MOVIES, 'user_id': user_id}


def get_movies(movies):  # TODO: организовать выдачу жанра, режиссёра и страны с помощью сложных запросов
    sess = session()
    movies = sess.query(Movie).filter(Movie.id.in_(movies)).all()
    sess.close()
    return movies


def descriptions_titles(offset):
    sess = session()
    titles = list(map(lambda x: x.title, sess.query(Movie.title).slice(offset, offset + 1000)))
    descriptions = list(map(lambda x: x.description, sess.query(Movie.description).slice(offset, offset + 1000)))
    sess.close()
    return titles, descriptions


def initialise_db():
    base.metadata.create_all(engine)

    df = pd.read_csv('~/PycharmProjects/smart-theater/data/movies_enhanced.csv', encoding='utf-8')
    df = df.drop(columns=["movieId"])
    start_time = time.monotonic()
    df.to_sql(con=engine, index=False, name=Movie.__tablename__, if_exists='append')
    del df
    df = pd.read_csv('~/PycharmProjects/smart-theater/data/ratings_enhanced.csv', encoding='utf-8')
    df = df[:358375]  # 2944075, 1476401
    n_users = len(df["user_id"].unique())
    print(n_users)
    # n_users = 3000
    users_data = [None] * n_users
    users_df = pd.DataFrame(data=users_data, columns=["username"])
    users_df.to_sql(con=engine, index=False, name=User.__tablename__, if_exists='append')
    df.to_sql(con=engine, index=False, name=Rating.__tablename__, if_exists='append', chunksize=200000)
    print("Elapsed time: ", time.monotonic() - start_time)


if __name__ == '__main__':
    initialise_db()
