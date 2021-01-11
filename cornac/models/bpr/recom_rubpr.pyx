# Copyright 2018 The Cornac Authors. All Rights Reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
# ============================================================================

# cython: language_level=3

cimport cython
from cython cimport floating, integral

from .recom_bpr import BPR
from .recom_bpr cimport RNGVector


class RUBPR(BPR):
    """Weighted Bayesian Personalized Ranking.

    Parameters
    ----------
    k: int, optional, default: 10
        The dimension of the latent factors.

    max_iter: int, optional, default: 100
        Maximum number of iterations or the number of epochs for SGD.

    learning_rate: float, optional, default: 0.001
        The learning rate for SGD.

    lambda_reg: float, optional, default: 0.001
        The regularization hyper-parameter.

    num_threads: int, optional, default: 0
        Number of parallel threads for training. If num_threads=0, all CPU cores will be utilized.
        If seed is not None, num_threads=1 to remove randomness from parallelization.

    trainable: boolean, optional, default: True
        When False, the model will not be re-trained, and input of pre-trained parameters are required.

    verbose: boolean, optional, default: True
        When True, some running logs are displayed.

    init_params: dictionary, optional, default: None
        Initial parameters, e.g., init_params = {'U': user_factors, 'V': item_factors, 'Bi': item_biases}.
        'V' and 'Bi' must be provided.

    seed: int, optional, default: None
        Random seed for weight initialization.
        If specified, training will take longer because of single-thread (no parallelization).

    References
    ----------
    * Gantner, Zeno, Lucas Drumond, Christoph Freudenthaler, and Lars Schmidt-Thieme. \
    "Personalized ranking for non-uniformly sampled items." In Proceedings of KDD Cup 2011, pp. 231-247. 2012.
    """

    def __init__(
        self, 
        name="RUBPR", 
        k=10, 
        max_iter=100, 
        learning_rate=0.001, 
        lambda_reg=0.01,
        num_threads=0, 
        trainable=True, 
        verbose=False, 
        init_params=None, 
        seed=None
    ):
        super().__init__(
            name=name, 
            k=k, 
            max_iter=max_iter, 
            learning_rate=learning_rate, 
            lambda_reg=lambda_reg, 
            num_threads=num_threads, 
            trainable=trainable, 
            verbose=verbose, 
            init_params=init_params, 
            seed=seed
        )

        assert self.i_factors is not None, "item factors should be provided"
        assert self.i_biases is not None, "item biases should be provided"

    @cython.cdivision(True)
    @cython.boundscheck(False)
    @cython.wraparound(False)
    def _fit_sgd(self, RNGVector rng_pos, RNGVector rng_neg, int num_threads,
                 integral[:] user_ids, integral[:] item_ids, 
                 integral[:] neg_item_ids, integral[:] indptr,
                 floating[:, :] U, floating[:, :] V, floating[:] B):
        """Fit the model parameters (U, V, B) with SGD
        """
        cdef:
            long num_samples = len(user_ids), s, i_index, j_index, correct = 0, skipped = 0
            long num_items = self.train_set.num_items
            integral f, i_id, j_id, thread_id
            floating z, score, temp

            floating lr = self.learning_rate
            floating reg = self.lambda_reg
            int factors = self.k

            floating * user
            floating * item_i
            floating * item_j

        with nogil, parallel(num_threads=num_threads):
            thread_id = get_thread_num()

            for s in prange(num_samples, schedule='guided'):
                i_index = rng_pos.generate(thread_id)
                i_id = item_ids[i_index]
                j_index = rng_neg.generate(thread_id)
                j_id = neg_item_ids[j_index]

                # if the user has liked the item j, skip this for now
                if has_non_zero(indptr, item_ids, user_ids[i_index], j_id):
                    skipped += 1
                    continue

                # get pointers to the relevant factors
                user, item_i, item_j = &U[user_ids[i_index], 0], &V[i_id, 0], &V[j_id, 0]

                # compute the score
                score = B[i_id] - B[j_id]
                for f in range(factors):
                    score = score + user[f] * (item_i[f] - item_j[f])
                z = 1.0 / (1.0 + exp(score))

                if z < .5:
                    correct += 1

                # update only the user factors via sgd.
                for f in range(factors):
                    temp = user[f]
                    user[f] += lr * (z * (item_i[f] - item_j[f]) - reg * user[f])

        return correct, skipped