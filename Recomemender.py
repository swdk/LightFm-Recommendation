import numpy as np
from config import fetch_movielens
from lightfm import LightFM
from lightfm.evaluation import precision_at_k
from lightfm.evaluation import auc_score

# fetch data and format it
data = fetch_movielens(min_rating=0)


for key, value in data.items():
    print(key, type(value), value.shape)

# print training and testing data
print(repr(data['train']))
print(repr(data['test']))

# CHALLENGE part 2 of 3 - use 3 different loss functions (so 3 different models), compare results, print results for
# the best one. - Available loss functions are warp, logistic, bpr, and warp-kos.

# create model
model = LightFM(learning_rate=0.02,loss='warp')
# train model
model.fit(data['train'], epochs=30, num_threads=2,item_features=data['item_features'],verbose=True)
# print(precision_at_k(model, data['test'], k=5,item_features=data['item_features']).mean())
print(auc_score(model,data['test'],data['train'],item_features=data['item_features']).mean())
print(precision_at_k(model,data['test'],data['train'],k=5,item_features=data['item_features']).mean())

# n_users, n_items = data['train'].shape
#
# train = np.zeros(shape = (n_users, n_items))
#
# for user_id in range(0,n_users):
# #     # print(user_id)
#    scores = model.predict(user_id, np.arange(n_items),item_features=data['item_features'])
#    train[user_id-1] = scores
#
# test=data['test'].todense()
# test = np.where(test!=0,test,np.nan)
# test = np.asmatrix(test)
# # np.set_printoptions(threshold=1000000000)
# print(np.sqrt(np.nanmean(np.square(train-test))))



def sample_recommendation(model, data, user_ids):
    # number of users and movies in training data
    n_users, n_items = data['train'].shape

    # generate recommendations for each user we input
    for user_id in user_ids:

        # movies they already like
        known_positives = data['item_labels'][data['train'].tocsr()[user_id].indices]

        # movies our model predicts they will like
        scores = model.predict(user_id, np.arange(n_items),item_features=data['item_features'])
        # rank them in order of most liked to least
        top_items = data['item_labels'][np.argsort(-scores)]

        # print out the results
        print("User %s" % user_id)
        print("     Known positives:")

        for x in known_positives[:3]:
            print("        %s" % x)

        print("     Recommended:")

        for x in top_items[:3]:
            print("        %s" % x)


sample_recommendation(model, data, [3, 25, 450])