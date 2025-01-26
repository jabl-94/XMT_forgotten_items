import numpy as np

from sklearn.decomposition import NMF as SKNMF

from forgotten_items.imports.utilities.data_management import *
from forgotten_items.imports.evaluation.evaluation_measures import *
from forgotten_items.imports.evaluation.calculate_aggregate_statistics import calculate_aggregate

class NMF:

    def __init__(self, n_user, n_item, n_factor, solver='cd', beta_loss='frobenius', tol=0.0001, max_iter=200,
                 random_state=None, alpha_W=0.0, alpha_H='same', l1_ratio=0.0, verbose=0, shuffle=False):
        self.__state = 'initialized'

        self.n_users = n_user
        self.n_items = n_item
        self.n_factor = n_factor

        self.alpha_W = alpha_W
        self.alpha_H = alpha_H
        self.l1_ratio = l1_ratio
        self.beta = beta_loss
        self.max_iter = max_iter
        self.tol = tol
        self.random_state = random_state
        self.l1_ratio = l1_ratio
        self.verbose = verbose
        self.shuffle = shuffle

        self.n_users = None
        self.n_items = None
        self.V = None
        self.W = None
        self.H = None
        self.R = None

    def get_state(self):
        return self.__state

    def __buildV(self, baskets, use_probabilities=False):

        user_count, item_count, users_item_count = count_users_items(baskets)

        self.n_users = len(user_count)
        self.n_items = len(item_count)

        self.V = np.zeros((self.n_users, self.n_items))
        for u in range(self.n_users):
            for i in range(self.n_items):
                r = users_item_count[u].get(i, 0.0)
                if use_probabilities:
                    r /= user_count[u]
                self.V[u, i] = r

    def build_model(self, baskets, use_probabilities=False):
        # print 'build V'
        self.__buildV(baskets, use_probabilities)
        # print 'density', 1.0 * len(self.V.nonzero()[0]) / (self.V.shape[0] * self.V.shape[1])

        sknmf = SKNMF(n_components=self.n_factor, init='random', solver='cd', beta_loss=self.beta, tol=self.tol,
                      max_iter=self.max_iter, alpha_W=self.alpha_W, alpha_H=self.alpha_H, l1_ratio=self.l1_ratio,
                      random_state=self.random_state, verbose=self.verbose, shuffle=self.shuffle)

        self.W = sknmf.fit_transform(self.V)
        self.H = sknmf.components_
        self.R = np.dot(self.W, self.H)

        self.__state = 'built'

        return self

    def update_model(self, new_baskets, use_probabilities=False):
        self.build_model(new_baskets, use_probabilities)
        return self

    def predict(self, user_id, pred_length=5):
        if self.__state != 'built':
            raise Exception('Model not built, prediction not available')

        # scores = list(self.R[user_id, :].A1)
        scores = list(self.R[user_id, :])
        item_rank = {k: v for k, v in enumerate(scores)}

        max_nbr_item = min(pred_length, len(item_rank))
        pred_basket = sorted(item_rank, key=item_rank.get, reverse=True)[:max_nbr_item]

        return pred_basket