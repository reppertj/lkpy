import logging
from itertools import starmap

import numpy as np
import pandas as pd
import scipy.sparse as sps


from ..matrix import sparse_ratings
from .mf_common import MFPredictor

_logger = logging.getLogger(__name__)


class LightFM(MFPredictor):
    """Lenskit interface to :py:mod:`LightFM.lightfm`

    Arguments:
        args:
        **kwargs: Arguments passed to :py:class:`lightfm.LightFM` constructor.
    Attributes:
        delegate(lightfm.LightFM):
            The :py:mod:`lightfm.LightFM` delegate.
        matrix_(scipy.sparse.coo_matrix):
            The user-item interactions matrix.
        user_index_(pandas.Index):
            The user index.
        item_idx_(pandas.Index):
            The item index.
    """

    def __init__(
        self, user_identity_features=True, item_identity_features=True, *args, **kwargs
    ):
        super().__init__()

        import lightfm

        self.delegate = lightfm.LightFM(*args, **kwargs)
        self._dataset = LFMDataset(
            user_identity_features=user_identity_features,
            item_identity_features=item_identity_features,
        )
        self._reset_state()

    @property
    def n_features(self):
        "The number of features."
        return self.delegate.no_components

    @property
    def n_users(self):
        "The number of users."
        return len(self.user_index_)

    @property
    def n_items(self):
        "The number of items."
        return len(self.item_index_)

    def lookup_user(self, user):
        """
        Look up the index for a user.

        Args:
            user: the user ID to look up

        Returns:
            int: the user index.
        """
        try:
            return self.user_index_.get_loc(user)
        except KeyError:
            return -1

    def lookup_items(self, items):
        """
        Look up the indices for a set of items.

        Args:
            items(array-like): the item IDs to look up.

        Returns:
            numpy.ndarray: the item indices. Unknown items will have negative indices.
        """
        return self.item_index_.get_indexer(items)

    def score(self, user, items, u_features=None):
        """
        Score a set of items for a user. User and item parameters must be indices
        into the matrices.

        Args:
            user(int): the user index
            items(array-like of int): the item indices
            raw(bool): if ``True``, do return raw scores without biases added back.

        Returns:
            numpy.ndarray: the scores for the items.
        """

        # get user vector
        uv = self.user_features_[user, :] if u_features is None else u_features
        # get item matrix
        im = self.item_features_[items, :]
        rv = np.matmul(im, uv)
        assert rv.shape[0] == len(items)
        assert len(rv.shape) == 1

        return rv

    def score_by_ids(self, user, items, u_features=None):
        if u_features is None:
            uidx = self.lookup_user(user)
            if uidx < 0:
                _logger.debug('user %s not in model', user)
                return pd.Series(np.nan, index=items)
        else:
            uidx = None

        # get item index & limit to valid ones
        items = np.array(items)
        iidx = self.lookup_items(items)
        good = iidx >= 0
        good_items = items[good]
        good_iidx = iidx[good]

        # multiply
        _logger.debug('scoring %d items for user %s', len(good_items), user)
        rv = self.score(uidx, good_iidx, u_features)

        res = pd.Series(rv, index=good_items)
        res = res.reindex(items)
        return res


    def fit(
        self,
        ratings,
        sample_weight_col=None,
        user_features=None,
        item_features=None,
        normalize_user_features=True,
        normalize_item_features=True,
        **kwargs
    ):
        """[summary]

        Arguments:
            ratings {[type]} -- [description]

        Keyword Arguments:
            user_features {[type]} -- [description] (default: {None})
            item_features {[type]} -- [description] (default: {None})
            user_identity_features {bool} -- [description] (default: {True})
            item_identity_features {bool} -- [description] (default: {True})
            normalize_user_features {bool} -- [description] (default: {True})
            normalize_item_features {bool} -- [description] (default: {True})
        """
        self._reset_state()
        self.partial_fit(
            ratings,
            sample_weight_col=sample_weight_col,
            user_features=user_features,
            item_features=item_features,
            normalize_user_features=normalize_user_features,
            normalize_item_features=normalize_item_features,
            **kwargs
        )

    def partial_fit(
        self,
        ratings,
        sample_weight_col=None,
        user_features=None,
        item_features=None,
        normalize_user_features=True,
        normalize_item_features=True,
        **kwargs
    ):
        """[summary]

        Arguments:
            ratings {[type]} -- [description]

        Keyword Arguments:
            user_features {[type]} -- [description] (default: {None})
            item_features {[type]} -- [description] (default: {None})
            user_identity_features {bool} -- [description] (default: {True})
            item_identity_features {bool} -- [description] (default: {True})
            normalize_user_features {bool} -- [description] (default: {True})
            normalize_item_features {bool} -- [description] (default: {True})
            **kwargs -- arguments passed to :py:meth:`lightfm.LightFM.fit`.

        Raises:
            TypeError: [description]

        Returns:
            [type] -- [description]
        """
        self._dataset.partial_fit(
            ratings.user, ratings.item, user_features, item_features
        )

        ratings_mat = self._dataset.build_interactions(ratings.rating)

        if sample_weight_col:
            sample_weight_mat = self._dataset.build_weights(ratings, sample_weight_col)
        else:
            sample_weight_mat = None

        if user_features:
            user_features = self._dataset.build_user_features(
                user_features, normalize_user_features
            )
        if item_features:
            item_features = self._dataset.build_item_features(
                item_features, normalize_item_features
            )

        _logger.info("fitting LightFM model")
        self.delegate.fit_partial(
            ratings_mat,
            user_features=self.user_features_,
            item_features=self.item_features_,
            sample_weight=sample_weight_mat,
            **kwargs
        )

        return self

    def _reset_state(self):

        self.delegate._reset_state()
        self._dataset._reset_state()

        self.user_idx_ = None
        self.item_idx_ = None
        self.user_features_ = None
        self.item_features_ = None
        self.user_identities_ = None
        self.item_identities_ = None

    def predict_for_user(self, user, items, ratings):
        return super().predict_for_user(user, items, ratings=ratings)

    def predict(
        self, users, items, user_features=None, item_features=None, normalize_user_features=True, normalize_item_features=True, **kwargs
    ):
        """[summary]

        Arguments:
            user {[type]} -- [description]
            items {[type]} -- [description]
            ratings {[type]} -- [description]

        Returns:
            [type] -- [description] NaN for items not in model.
        """
        self._dataset.partial_fit(pd.Series([user]), items, user_features, item_features)

        uidx = self.dataset_.user_idx_.get_indexer(user)
        iidx = self.dataset_.item_idx_.get_indexer(items)


        # Limit to users/items in model

        good = uidx >= 0
        users = np.array(users)
        good_users = users[good]
        good_uidx = uidx[good]

        good = iidx >= 0
        items = np.array(items)
        good_items = items[good]
        good_iidx = iidx[good]

        ufeat_mat = self._dataset.build_user_features(user_features, normalize_user_features)
        ifeat_mat = self._dataset.build_item_features(item_features, normalize_item_features)

        scores = self.delegate.predict(uidx, good_iidx, **kwargs)
        res = pd.Series(scores, index=good_items)
        res = res.reindex(items)
        return res




class LFMDataset:
    """Maintains mappings between user ids, item ids, and feature ids and matrix indices and generates sparse matrices for use by LightFM.
    """

    def __init__(self, user_identity_features, item_identity_features):
        self.user_identity_features_ = user_identity_features
        self.item_identity_features_ = item_identity_features
        self._reset_state()

    def fit(self, users, items, user_features=None, item_features=None):
        self._reset_state()
        self.partial_fit(users, items, user_features, item_features)

    def partial_fit(
        self, users=None, items=None, user_features=None, item_features=None
    ):
        user_idx, item_idx, ufeat_idx, ifeat_idx = [None] * 4

        if users:
            users = [users, user_features.user] if user_features else [users]
            user_idx = pd.Index(np.unique(pd.concat(users)))
        if items:
            items = [items, item_features.item] if item_features else [items]
            item_idx = pd.Index(np.unique(pd.concat(items)))
        if user_features:
            ufeat_idx = pd.Index(np.unique(user_features.feature))
        if item_features:
            ifeat_idx = pd.Index(np.unique(item_features.feature))

        self.validate_features(ufeat_idx, ifeat_idx)

        self.add_to_idx(user_idx, item_idx, ufeat_idx, ifeat_idx)

        # We can only add identity features once
        if self.user_identity_features_ and self.ufeat_identity_idx is None:
            self.ufeat_identity_idx_ = self.user_idx_
        if self.item_identity_features_ and self.ifeat_identity_idx_ is None:
            self.ifeat_identity_idx_ = self.item_idx_

        return self

    def add_to_idx(self, user_idx=None, item_idx=None, ufeat_idx=None, ifeat_idx=None):
        if user_idx:
            self.user_idx_, prev_idx = user_idx, self.user_idx_
            if prev_idx:
                self.user_idx_ = prev_idx.append(self.user_idx_).drop_duplicates(
                    keep="first"
                )
        if item_idx:
            self.item_idx_, prev_idx = item_idx, self.item_idx_
            if prev_idx:
                self.item_idx_ = prev_idx.append(self.item_idx_).drop_duplicates(
                    keep="first"
                )
        if ufeat_idx:
            self.ufeat_idx_, prev_idx = ufeat_idx, self.ufeat_idx_
            if prev_idx:
                self.ufeat_idx = prev_idx.append(self.ufeat_idx_).drop_duplicates(
                    keep="first"
                )
        if ifeat_idx:
            self.ifeat_idx_, prev_idx = ifeat_idx, self.ifeat_idx_
            if prev_idx:
                self.ifeat_idx_ = prev_idx.append(self.ifeat_idx_).drop_duplicates(
                    keep="first"
                )

    def _reset_state(self):
        self.user_idx_ = None
        self.item_idx_ = None
        self.ifeat_idx_ = None
        self.ufeat_idx_ = None
        self.ufeat_identity_idx_ = None
        self.ifeat_identity_idx_ = None

    def _validate_features(self, ufeat_idx, ifeat_idx):
        if ufeat_idx and self.ufeat_idx_:
            diff = ufeat_idx.difference(self.ufeat_idx_)
            if len(diff) != 0:
                raise ValueError(
                    "User feature %s not in mapping. Call fit first.", diff[0]
                )
        if ifeat_idx and self.ifeat_idx_:
            diff = ifeat_idx.difference(self.ifeat_idx_)
            if len(diff) != 0:
                raise ValueError(
                    "Item feature %s not in mapping. Call fit first.", diff[0]
                )

    def build_interactions(self, ratings):
        ratings_mat, self.user_idx_, self.item_idx_ = sparse_ratings(
            ratings, scipy="coo", users=self.user_idx_, items=self.item_idx_
        )
        return ratings_mat

    def build_weights(self, ratings, weight_col):
        row_ind = self.user_idx_.get_indexer(ratings.user).astype(np.intc)
        if np.any(row_ind < 0):
            raise ValueError("Users not in mapping. Call fit or partial_fit first.")
        col_ind = self.item_idx_.get_indexer(ratings.item).astype(np.intc)
        if np.any(col_ind < 0):
            raise ValueError("Items not in mapping. Call fit or partial_fit first.")

        if weight_col not in ratings.columns:
            raise ValueError("Weight column %s not in ratings", weight_col)
        else:
            vals = np.require(ratings[weight_col].values, np.float32)

        weights_mat = sps.coo_matrix((vals, (row_ind, col_ind)), shape=self.get_shape())

        return weights_mat

    def get_shape(self):
        if self._user_idx_ and self.item_idx_:
            return (len(self.user_idx_), len(self.item_idx_))

    def build_user_features(self, user_features, normalize):
        if not {"user", "feature"} <= set(user_features.columns):
            raise ValueError(
                "Features must be dataframe of (user, feature) or (user, feature, value)"
            )
        else:
            row_ind = self.user_idx_.get_indexer(user_features.feature).astype(np.intc)
            col_ind = self.ufeat_idx_.get_indexer(user_features.feature).astype(np.intc)

            if np.any(row_ind < 0):
                raise ValueError("users in features not in dataset")
            if np.any(col_ind < 0):
                raise ValueError("features in user features not dataset")

            shape = (len(self.user_idx_), len(self.ufeat_idx_))

            if "value" in user_features.columns:
                data = np.require(user_features.value, dtype=np.float32)
            else:
                data = np.ones_like(col_ind, dtype=np.float32)

            identity_idx = self.ufeat_identity_idx_

            return self._build_features_mat_(
                data, row_ind, col_ind, shape, normalize, identity_idx
            )

    def build_item_features(self, item_features, normalize):
        if not {"item", "feature"} <= set(item_features.columns):
            raise ValueError(
                "Features must be dataframe of (item, feature) or (item, feature, value)"
            )
        else:
            row_ind = self.item_idx_.get_indexer(item_features.feature).astype(np.intc)
            col_ind = self.ufeat_idx_.get_indexer(item_features.feature).astype(np.intc)

            if np.any(row_ind < 0):
                raise ValueError("items in features not in dataset")
            if np.any(col_ind < 0):
                raise ValueError("features in item features not dataset")

            shape = (len(self.user_idx_), len(self.ufeat_idx_))

            if "value" in item_features.columns:
                data = np.require(item_features.value, dtype=np.float32)
            else:
                data = np.ones_like(col_ind, dtype=np.float32)

            identity_idx = self.ifeat_identity_idx_

            return self._build_features_mat(
                data, row_ind, col_ind, shape, normalize, identity_idx
            )

    def _build_features_mat(
        data, row_ind, col_ind, shape, normalize, identity_idx=None
    ):
        mat = sps.csr_matrix((data, (row_ind, col_ind)), shape=shape)

        if identity_idx:
            eye = sps.identity(len(identity_idx), format="csr", dtype=np.float32)
            other = sps.csr_matrix(
                (shape[0] - len(identity_idx), shape[1]), dtype=np.float32
            )
            identity = sps.vstack([eye, other], format="csr")
            mat = sps.hstack([mat, identity], format="csr")

        if normalize:
            if np.any(mat.getnnz(1) == 0):
                raise ValueError("Cannot normalize: Some rows have zero norm")
            from sklearn.preprocessing import normalize

            mat = normalize(mat, norm="l1", copy=False)

        return mat

    def mapping(self):
        return starmap(
            self._build_mapping,
            (
                (self.user_idx_, "user"),
                (self.item_idx_, "item"),
                (self.ufeat_idx_, "user feature"),
                (self.ifeat_idx_, "item feature"),
            ),
        )

    @staticmethod
    def _build_mapping(idx, name):
        return pd.Series(np.arange(len(idx), index=idx, name=name).to_dict())

