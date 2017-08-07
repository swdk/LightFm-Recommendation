import numpy as np
from config import fetch_movielens
from lightfm import LightFM
import pickle

model = pickle.load(open("model/model.dat", "rb"))
data = pickle.load(open("model/data.dat", "rb"))

def sample_recommendation(model, data, user_ids):
    # number of users and movies in training data
    n_users, n_items = data['train'].shape
    # load model


    # generate recommendations for each user we input
    for user_id in user_ids:

        # movies they already like
        known_positives = data['item_labels'][data['train'].tocsr()[user_id].indices]

        # movies our model predicts they will like
        scores = model.predict(user_id, np.arange(n_items),item_features=data['item_features'],user_features=data['user_features'])
        # rank them in order of most liked to least
        top_items = data['item_labels'][np.argsort(-scores)]

        # print out the results
        print("User %s" % user_id)
        print("     Known positives:")

        for x in known_positives[:5]:
            print("        %s" % x)

        print("     Recommended:")

        for x in top_items[:5]:
            print("        %s" % x)


sample_recommendation(model, data, [300, 215, 40])