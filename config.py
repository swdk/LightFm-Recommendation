import itertools
import zipfile

import numpy as np

import scipy.sparse as sp

from lightfm.datasets import _common


def _read_raw_data(path):
    """
    Return the raw lines of the train and test files.
    """

    with zipfile.ZipFile(path) as datafile:
        return (datafile.read('ml-100k/ua.base').decode().split('\n'),
                datafile.read('ml-100k/ua.test').decode().split('\n'),
                datafile.read('ml-100k/u.item').decode(errors='ignore').split('\n'),
                datafile.read('ml-100k/u.genre').decode(errors='ignore').split('\n'),
                datafile.read('ml-100k/u.user').decode(errors='ignore').split('\n'))


def _parse(data):

    for line in data:

        if not line:
            continue

        uid, iid, rating, timestamp = [int(x) for x in line.split('\t')]

        # Subtract one from ids to shift
        # to zero-based indexing
        yield uid - 1, iid - 1, rating, timestamp


def _get_dimensions(train_data, test_data):

    uids = set()
    iids = set()

    for uid, iid, _, _ in itertools.chain(train_data,
                                          test_data):
        uids.add(uid)
        iids.add(iid)

    rows = max(uids) + 1
    cols = max(iids) + 1

    return rows, cols


def _build_interaction_matrix(rows, cols, data, min_rating):

    mat = sp.lil_matrix((rows, cols), dtype=np.int32)

    for uid, iid, rating, _ in data:
        if rating >= min_rating:
            mat[uid, iid] = rating



    # return mat.tocoo()
    return mat

def _parse_item_metadata(num_items,num_users,
                         item_metadata_raw, genres_raw,user_raw):

    genres = []

    for line in genres_raw:
        if line:
            genre, gid = line.split('|')
            # print('genre:{}'.format(genre))
            genres.append('genre:{}'.format(genre))

    id_feature_labels = np.empty(num_items, dtype=np.object)
    genre_feature_labels = np.array(genres)

    id_features = sp.identity(num_items,
                              format='csr',
                              dtype=np.float32)
    genre_features = sp.lil_matrix((num_items, len(genres)),
                                   dtype=np.float32)

    for line in item_metadata_raw:

        if not line:
            continue

        splt = line.split('|')

        # Zero-based indexing
        iid = int(splt[0]) - 1
        title = splt[1]

        id_feature_labels[iid] = title

        item_genres = [idx for idx, val in
                       enumerate(splt[5:])
                       if int(val) > 0]

        for gid in item_genres:
            genre_features[iid, gid] = 1.0

    return (id_features, id_feature_labels,
            genre_features.tocsr(), genre_feature_labels)

def _parse_item_metadata2(num_users,item_raw):
    # 22 occupations
    # user_occupation = sp.lil_matrix((num_users, 22),
    #                                dtype=np.float32)
    user_age = sp.lil_matrix((num_users, 16),
                             dtype=np.float32)
    user_gender = sp.lil_matrix((num_users, 3),
                             dtype=np.float32)

    # 16 age groups
    # 2 gender

    max_age_group = 0
    for line in item_raw:
        if not line:
            continue

        splt = line.split('|')
        uid = int(splt[0])
        age = int(splt[1])
        age_group= round(age/5)
        user_age[uid, age_group] = 1
        gender = splt[2]
        if(gender == 'M'):
            user_gender[uid, 1] = 1
        else:
            user_gender[uid,2] = 1

    user_feature  = sp.hstack([user_age, user_gender])

    # print(user_feature)

    return(user_feature.tocsr())

def fetch_movielens(data_home=None, indicator_features=True, genre_features=True,
                    min_rating=0.0, download_if_missing=True):
    """
    Fetch the `Movielens 100k dataset <http://grouplens.org/datasets/movielens/100k/>`_.
    The dataset contains 100,000 interactions from 1000 users on 1700 movies,
    and is exhaustively described in its
    `README <http://files.grouplens.org/datasets/movielens/ml-100k-README.txt>`_.
    Parameters
    ----------
    data_home: path, optional
        Path to the directory in which the downloaded data should be placed.
        Defaults to ``~/lightfm_data/``.
    indicator_features: bool, optional
        Use an [n_users, n_users] identity matrix for item features. When True with genre_features,
        indicator and genre features are concatenated into a single feature matrix of shape
        [n_users, n_users + n_genres].
    genre_features: bool, optional
        Use a [n_users, n_genres] matrix for item features. When True with item_indicator_features,
        indicator and genre features are concatenated into a single feature matrix of shape
        [n_users, n_users + n_genres].
    min_rating: float, optional
        Minimum rating to include in the interaction matrix.
    download_if_missing: bool, optional
        Download the data if not present. Raises an IOError if False and data is missing.
    Notes
    -----
    The return value is a dictionary containing the following keys:
    Returns
    -------
    train: sp.coo_matrix of shape [n_users, n_items]
         Contains training set interactions.
    test: sp.coo_matrix of shape [n_users, n_items]
         Contains testing set interactions.
    item_features: sp.csr_matrix of shape [n_items, n_item_features]
         Contains item features.
    item_feature_labels: np.array of strings of shape [n_item_features,]
         Labels of item features.
    item_labels: np.array of strings of shape [n_items,]
         Items' titles.
    """

    if not (indicator_features or genre_features):
        raise ValueError('At least one of item_indicator_features '
                         'or genre_features must be True')

    zip_path = _common.get_data(data_home,
                                ('https://github.com/maciejkula/'
                                 'lightfm_datasets/releases/'
                                 'download/v0.1.0/movielens.zip'),
                                'movielens100k',
                                'movielens.zip',
                                download_if_missing)

    # Load raw data
    (train_raw, test_raw,
     item_metadata_raw, genres_raw,user_raw) = _read_raw_data(zip_path)


    # Figure out the dimensions
    num_users, num_items = _get_dimensions(_parse(train_raw),
                                           _parse(test_raw))



    # Load train interactions
    train = _build_interaction_matrix(num_users,
                                      num_items,
                                      _parse(train_raw),
                                      min_rating)


    # Load test interactions
    test = _build_interaction_matrix(num_users,
                                     num_items,
                                     _parse(test_raw),
                                     min_rating)

    assert train.shape == test.shape

    # Load metadata features
    (id_features, id_feature_labels,
     genre_features_matrix, genre_feature_labels) = _parse_item_metadata(num_items,num_users,
                                                                             item_metadata_raw,
                                                                            genres_raw,
                                                                            user_raw)
    (user_age) = _parse_item_metadata2(num_items,user_raw)
    # print(user_age)
    # assert user_age.shape ==(num_users,16)

    assert id_features.shape == (num_items, len(id_feature_labels))
    assert genre_features_matrix.shape == (num_items, len(genre_feature_labels))

    if indicator_features and not genre_features:
        features = id_features
        feature_labels = id_feature_labels
    elif genre_features and not indicator_features:
        features = genre_features_matrix
        feature_labels = genre_feature_labels
    else:
        features = sp.hstack([id_features, genre_features_matrix]).tocsr()
        feature_labels = np.concatenate((id_feature_labels,
                                         genre_feature_labels))

    data = {'train': train,
            'test': test,
            'item_features': features,
            'item_feature_labels': feature_labels,
            'item_labels': id_feature_labels,
            'user_features':user_age}

    return data