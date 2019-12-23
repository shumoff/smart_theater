import os
import sys
project_dirname = os.path.abspath(__file__ + '/../../')
sys.path.append(project_dirname)

import flask
import json
import config
from core import algorithms
from backend import database
from flask_cors import CORS, cross_origin
from flask import Flask
from flask import jsonify
from flask import request
app = Flask(__name__)
cors = CORS(app)
app.config['CORS_HEADERS'] = 'Content-Type'


@app.route('/registration', methods=['POST'])
@cross_origin()
def register_user():
    query = json.loads(request.data)
    email = query['email']
    username = query['username']
    password = query['password']
    db_resp = database.add_new_user(username, email, password)
    if db_resp['OK']:
        response = flask.make_response(jsonify(OK=db_resp['OK'], token=db_resp['token'], username=username))
        return response
    else:
        response = flask.make_response(jsonify(OK=db_resp['OK'], error=db_resp['error']))
    return response


@app.route('/login', methods=['POST'])
@cross_origin()
def login_user():
    query = json.loads(request.data)
    login = query['login']
    password = query['password']
    db_resp = database.check_logpass(login, password)
    if db_resp['OK']:
        token = database.renew_token(db_resp['username'])
        response = flask.make_response(jsonify(OK=db_resp['OK'], token=token, username=db_resp['username']))
    else:
        response = flask.make_response(jsonify(OK=db_resp['OK'], error=db_resp['error']))
    return response


@app.route('/logout', methods=['POST'])
@cross_origin()
def logout_user():
    query = json.loads(request.data)
    token = query['token']
    database.spoil_token(token)
    response = flask.make_response(jsonify(OK=True))
    return response


@app.route('/auth', methods=['POST'])
@cross_origin()
def auth_user():
    query = json.loads(request.data)
    token = query['token']
    db_resp = database.check_token(token)
    if db_resp['OK']:
        response = flask.make_response(jsonify(OK=db_resp['OK'], username=db_resp['username']))
    else:
        response = flask.make_response(jsonify(OK=db_resp['OK']))
    return response


@app.route('/profiling', methods=['POST'])
@cross_origin()
def profile_user():
    pass


@app.route('/search', methods=['POST'])
@cross_origin()
def search():
    query = json.loads(request.data)
    q = query['q']
    titles, descriptions = [1, 2], [1, 2]
    top_results = {}
    offset = 0
    while len(titles) > 1:
        titles, descriptions = database.descriptions_titles(offset)
        top_results.update(algorithms.search_movie(q, titles, descriptions, offset))
        offset += 1000
    top_results = [int(k) for k, v in sorted(top_results.items(), key=lambda x: x[1], reverse=True)]
    movies = database.get_movies(top_results[:config.MOVIES_TO_SEND])
    movie_json = []
    for movie in movies:
        movie_json.append({'movie_id': movie.id, 'title': movie.title, 'year': movie.year,
                           'runtime': movie.runtime, 'imdb_rating': movie.imdb_rating,
                           'description': movie.description, 'poster_link': movie.poster_link,
                           'link': movie.link})
    response = {'OK': True, 'movies': movie_json}
    return response


@app.route('/getRecommendations', methods=['POST'])
@cross_origin()
def get_recommendations(): # TODO: доделать - надо записывать в бд similarity и predictions
    query = json.loads(request.data)
    type_ = query['type']
    if type_ == 'personal':
        token = query['token']
        db_resp = database.check_token(token)
        user_status = database.user_status(db_resp['username'])
        if user_status['status']:
            data = database.get_ratings_matrix()
            predictions, similarity = algorithms.predict_user(user_status['user_id'], data['users_amount'],
                                                              data['movies_amount'], data['data'])
            movies = database.get_movies(
                list(predictions.sort_values(by=['rating'], ascending=False)['movie_id'].iloc[:config.MOVIES_TO_SEND]))
            movie_json = []
            for movie in movies:
                movie_json.append({'movie_id': movie.id, 'title': movie.title, 'year': movie.year,
                                   'runtime': movie.runtime, 'imdb_rating': movie.imdb_rating,
                                   'description': movie.description, 'poster_link': movie.poster_link,
                                   'link': movie.link})
            response = {'OK': True, 'movies': movie_json}
            return response
        else:
            response = flask.make_response(jsonify(OK=user_status['status']))
            return response

    elif type_ == 'similar':
        movie_id = query['movie_id']
        data = database.get_ratings_matrix()
        similarity = algorithms.similar_movies(movie_id, data['users_amount'], data['movies_amount'], data['data'])
        movies = database.get_movies(
            list(similarity.sort_values(
                by=['sim_coef'], ascending=False)['other_movie_id'].iloc[1:config.MOVIES_TO_SEND+1]))
        movie_json = []
        for movie in movies:
            movie_json.append({'movie_id': movie.id, 'title': movie.title, 'year': movie.year,
                               'runtime': movie.runtime, 'imdb_rating': movie.imdb_rating,
                               'description': movie.description, 'poster_link': movie.poster_link,
                               'link': movie.link})
        response = {'OK': True, 'movies': movie_json}
        return response

    else:  # TODO: гибрид
        pass
