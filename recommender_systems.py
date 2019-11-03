import numpy as np
import scipy
from lightfm import LightFM
# LightFM's built-in Dataset class to build an interaction dataset from raw data
from lightfm.datasets import fetch_movielens 

# data is a sparse matrix
data = fetch_movielens(min_rating=4.0)

model = LightFM(no_components=30, loss='warp')

print(repr(data['train']))
print(repr(data['test']))

# Model training is accomplished via SGD (stochastic gradient descent). This means that for every pass 
# through the data - an epoch or run - the model learns to fit the data more and more closely. 
# We will run it for 10 epochs in this example. We can also run it on multiple cores, so we will set that to 2.
model.fit(data['train'], epochs=30, num_threads=2)
# print(dir(data['train']))
# print(data['item_labels'][data['train'].tocsr()[10].indices])

def sample_recommendation(model, data, user_ids):
	n_users, n_items = data['train'].shape

	for user_id in user_ids:
		# movies users already like
		known_positives = data['item_labels'][data['train'].tocsr()[user_id].indices]

		# movies they will like
        scores = model.predict(user_id, np.arange(n_items))

        # arrange movies in descending order of their liking - most liked to least
        top_items = data['item_labels'][np.argsort(-scores)]

        print("User %s" % user_id)
        print("     Known positives:")

        for x in known_positives[:3]:
            print("        %s" % x)

        print("     Recommended:")

        for x in top_items[:3]:
            print("        %s" % x)


sample_recommendation(model, data, [10,35,200])