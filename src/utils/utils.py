import pickle
import os
import pandas as pd
from sklearn.metrics.pairwise import cosine_similarity
import scipy.sparse
import numpy as np
import sklearn


class utils():
    def __init__(self, user_id):
        self.root_dir = os.path.join(os.path.dirname(__file__), "..")
        self.data_path = self.root_dir + "/data/"
        self.model_path = self.root_dir + "/model/"

        self.user_id = user_id

        self.df_transaction, self.df_products = self.read_data()
        self.df_chart = self.create_interaction_chart()
        self.tfidf_matrix, self.tfidf_feature, self.item_ids = self.read_product_profile()


    def read_data(self):
        df_transaction = pd.read_csv(self.data_path + 'recommendations_take_home.csv')
        df_transaction = df_transaction[df_transaction['is_gift'] != True]
        df_products = pd.read_csv(self.data_path + 'products.csv')
        return df_transaction, df_products

    def read_product_profile(self):
        tfidf_matrix = pickle.load(open(self.model_path + "tfidf_matrix.pickle", "rb"))
        tfidf_feature = pickle.load(open(self.model_path + "tfidf_feature.pickle", "rb"))
        item_ids = pickle.load(open(self.model_path + "item_ids.pickle", "rb"))
        return tfidf_matrix, tfidf_feature, item_ids

    def create_interaction_chart(self):
        df_chart = self.df_transaction.groupby(['user_id_hash', 'catalog_item_id'])['quantity']\
            .sum()\
            .to_frame()\
            .reset_index()\
            .rename(columns={'quantity': '#total purchased quantity'})\
            .set_index('user_id_hash')
        return df_chart

    def get_items_interacted(self):
        interacted_items = self.df_chart.loc[self.user_id]['catalog_item_id']
        return set(interacted_items if type(interacted_items) == pd.Series else [interacted_items])

    def get_item_profile(self, item_id):
        idx = self.item_ids.index(item_id)
        item_profile = self.tfidf_matrix[idx:idx+1]
        return item_profile

    def get_item_profiles(self, ids):
        item_profiles_list = [self.get_item_profile(x) for x in ids]
        item_profiles = scipy.sparse.vstack(item_profiles_list)
        return item_profiles

    def build_users_profile(self):
        # get a list of purchased product for each user
        ids = self.get_items_interacted()
        # get tf-idf matrix for each product for each user
        user_product_vector = self.get_item_profiles(ids)
        # weighted average by the number of units purchased
        weights = np.array(self.df_chart.loc[self.user_id]['#total purchased quantity']).reshape(-1, 1)
        user_product_vector_avg = np.sum(user_product_vector.multiply(weights), axis=0) / np.sum(weights)
        user_product_norm = sklearn.preprocessing.normalize(user_product_vector_avg)
        return user_product_norm

    def generate_recommendation(self, topn=20):
        user_profile = self.build_users_profile()
        # calculate cosine similarity between each user's profile and each product
        cosine_similarities = cosine_similarity(user_profile, self.tfidf_matrix)
        # select top n items to form a recommendation list
        similar_indices = cosine_similarities.argsort().flatten()[-topn:]
        similar_items = sorted([(self.item_ids[i], cosine_similarities[0, i]) for i in similar_indices],
                               key=lambda x: -x[1])
        # ignore the items that user has already purchased in the recommendation list
        items_to_ignore = self.get_items_interacted()
        similar_items_filtered = list(filter(lambda x: x[0] not in items_to_ignore, similar_items))
        recommendations_df = pd.DataFrame(similar_items_filtered, columns=['catalog_item_id', 'score'])
        recommendations_df = pd.merge(recommendations_df,
                                      self.df_products[['catalog_item_id', 'catalog_item_name', 'brand_name']],
                                      on='catalog_item_id', how='left')
        # convert an output into a dictionary
        recommend_list = []
        for index, row in recommendations_df.iterrows():
            recommend_list.append(row.to_dict())
        rc = {}
        rc[self.user_id] = recommend_list
        return rc

    def recommend_most_popular(self):
        most_popular = pickle.load(open(self.model_path + "most_popular_list.pickle", "rb"))
        rc = {}
        rc[self.user_id] = most_popular
        return rc

    def model(self):
        if self.df_chart.index.str.contains(self.user_id).sum() > 0:
            rc = self.generate_recommendation()
        else:
            rc = self.recommend_most_popular()
        return rc


# user_id = '000a984b1f8df5dc7d08ff18f7771594'
#
# a = utils(user_id)
# rc = a.model()
# print(a.get_items_interacted())
# print(rc)


