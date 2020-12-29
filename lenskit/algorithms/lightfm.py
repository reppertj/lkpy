import logging

import numpy as np
import pandas as pd

from ..matrix import sparse_ratings
from . import Predictor

_logger = logging.getLogger(__name__)


class LightFM(Predictor):
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
        item_index_(pandas.Index):
            The item index.
    """

    def __init__(self, *args, **kwargs):
        super().__init__()

        import lightfm

        self.delegate = lightfm.LightFM(*args, **kwargs)

    def fit(
        self,
        ratings,
        user_features=None,
        item_features=None,
        user_identity_features=True,
        item_identity_features=True,
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
        if isinstance(ratings, pd.DataFrame):
            ratings, users, items = sparse_ratings(ratings, scipy=True)
            ratings = ratings.tocoo()  # TODO: Change after sparse_ratings refactor
        elif isinstance(ratings, scipy.sparse.coo_matrix):
            n_users, n_items = ratings.shape
            users = pd.Index(np.arange(n_users), name="user")
            items = pd.Index(np.arange(n_items), name="item")
        else:
            raise TypeError("unsupported type %s for ratings", type(ratings))

        self.user_index_ = users
        self.item_index_ = items

        if isinstance(user_features, pd.DataFrame):
            if set(user_features.columns) - {'user', 'feature'} in [set(), {'value'}]:
                user_features_builder = self._build_features_long
            else:
                user_features_builder = self._build_features_wide
            self.user_features_ = user_features_builder(
                users,
                "user",
                user_features,
                user_identity_features,
                normalize_user_features,
            )
        elif user_features is None or isinstance(
            user_features, scipy.sparse.coo_matrix
        ):
            self.user_features_ = user_features
        else:
            raise TypeError(
                "unsupported type %s for user features", type(user_features)
            )

        if isinstance(item_features, pd.DataFrame):
            if set(item_features.columns) - {'item', 'feature'} in [set(), {'value'}]:
                item_features_builder = self._build_features_long
            else:
                item_features_builder = self._build_features_wide
            self.item_features_ = item_features_builder(
                items,
                "item",
                item_features,
                item_identity_features,
                normalize_item_features,
            )
        elif item_features is None or isinstance(
            item_features, scipy.sparse.coo_matrix
        ):
            self.item_features_ = item_features
        else:
            raise TypeError(
                "unsupported type %s for item features", type(item_features)
            )

        _logger.info("fitting LightFM model")
        self.delegate.fit(ratings, **kwargs)

        return self

    def predict_for_user(self, user, items, ratings=None, **kwargs):
        """[summary]

        Arguments:
            user {[type]} -- [description]
            items {[type]} -- [description]
            ratings {[type]} -- [description]

        Returns:
            [type] -- [description] NaN for items not in model.
        """
        uidx = self.user_index_.get_loc(user)
        iidx = self.item_index_.get_indexer(items)

        good = iidx >= 0  # Limit to items in model
        items = np.array(items)
        good_items = items[good]
        good_iidx = iidx[good]

        scores = self.delegate.predict(uidx, good_iidx, **kwargs)
        res = pd.Series(scores, index=good_items)
        res = res.reindex(items)
        return res

    @staticmethod
    def _build_features_wide(idx, idx_col, features, identity_features, normalize):
        n_rows = len(idx)
        n_features = len(features.columns) - 1

        row_indexer = idx.get_indexer(features[idx_col])
        if np.any(row_indexer < 0):
            raise ValueError("Entry in features dataframe missing in ratings")
        n_entries = len(row_indexer)

        row_ind = np.tile(row_indexer, n_features)
        col_ind = np.repeat(np.arange(n_features), n_entries)
        data = (
            features.drop(columns=[idx_col]).to_numpy(dtype=np.float32).ravel(order="F")
        )

        mat = LightFM._build_features(
            data, row_ind, col_ind, n_rows, n_features, identity_features, normalize
        )

        return mat

    @staticmethod
    def _build_features_long(idx, idx_col, features, identity_features, normalize):
        n_rows = len(idx)
        feat_idx = pd.Index(np.unique(features.feature), name="feature")
        n_features = len(feat_idx)

        row_ind = idx.get_indexer(features[idx_col])

        if np.any(row_ind < 0):
            raise ValueError("Entry in features dataframe missing in ratings")
        col_ind = feat_idx.get_indexer(features.feature).astype(np.intc)

        if "value" in features.columns:
            data = np.require(ratings.rating.values, np.float32)
        else:
            data = np.ones_like(col_ind, dtype=np.float32)

        mat = LightFM._build_features(
            data, row_ind, col_ind, n_rows, n_features, identity_features, normalize
        )
        return mat

    @staticmethod
    def _build_features(
        data, row_ind, col_ind, n_rows, n_features, identity_features, normalize
    ):
        import scipy.sparse as sps

        mat = sps.csr_matrix((data, (row_ind, col_ind)), shape=(n_rows, n_features))

        if identity_features:
            mat = sps.hstack([mat, sps.eye(n_rows, format="csr", dtype=np.float32)])

        if normalize:
            if np.any(mat.getnnz(1) == 0):
                raise ValueError("Cannot normalize: Some rows have zero norm")
            from sklearn.preprocessing import normalize

            mat = normalize(mat, norm="l1", copy=False)

        return mat
