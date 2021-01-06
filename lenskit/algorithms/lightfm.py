import logging

import numpy as np
import pandas as pd
import scipy.sparse as sps


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

        import lightfm

        super().__init__()
        self.delegate = lightfm.LightFM(*args, **kwargs)
        self._reset_state()

    def fit(
        self,
        ratings,
        sample_weight_col=None,
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
        """
        self._reset_state()
        self.partial_fit(
            ratings,
            sample_weight_col=sample_weight_col,
            user_features=user_features,
            item_features=item_features,
            user_identity_features=user_identity_features,
            item_identity_features=item_identity_features,
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
        self._add_to_index(users_to_add=[ratings.user])
        self._add_to_index(items_to_add=[ratings.item])

        if user_features is not None:
            self.user_features_ = self._process_user_features(
                user_features, user_identity_features, normalize_user_features
            )

        if item_features is not None:
            self.item_features_ = self._process_item_features(
                item_features, item_identity_features, normalize_item_features
            )

        ratings_mat, self.user_index_, self.item_index_ = sparse_ratings(
            ratings, scipy="coo", users=self.user_index_, items=self.item_index_
        )

        if sample_weight_col:
            sample_weight = self._build_weights(ratings, sample_weight_col)
        else:
            sample_weight = None

        _logger.info("fitting LightFM model")
        self.delegate.fit_partial(
            ratings_mat,
            user_features=self.user_features_,
            item_features=self.item_features_,
            sample_weight=sample_weight,
            **kwargs
        )

        return self

    def _process_user_features(
        self, user_features, user_identity_features, normalize_user_features
    ):
        if set(user_features.columns) - {"user", "feature"} not in (set(), {"value"}):
            raise ValueError(
                "Features must be dataframe of (user, feature) or (user, feature, value)"
            )
        else:
            self._add_to_index(users_to_add=[user_features.user])
            user_features, self.n_user_features = self._build_features(
                self.user_index_,
                "user",
                user_features,
                user_identity_features,
                normalize_user_features,
            )
        return user_features

    def _process_item_features(
        self, item_features, item_identity_features, normalize_item_features
    ):
        if set(item_features.columns) - {"item", "feature"} not in (set(), {"value"}):
            raise ValueError(
                "Features must be dataframe of (item, feature) or (item, feature, value)"
            )
        else:
            self._add_to_index(items_to_add=[item_features.item])
            item_features, self.n_item_features = self._build_features(
                self.item_index_,
                "item",
                item_features,
                item_identity_features,
                normalize_item_features,
            )
        return item_features

    def _reset_state(self):

        self.delegate._reset_state()

        self.user_index_ = None
        self.item_index_ = None
        self.user_features_ = None
        self.item_features_ = None
        self.user_identities_ = None
        self.item_identities_ = None

    def _build_weights(self, ratings, sample_weight_col):
        row_ind = self.user_index_.get_indexer(ratings.user)
        col_ind = self.item_index_.get_indexer(ratings.item)

        weights = np.require(ratings[sample_weight_col], np.float32)

        n_rows, n_cols = len(self.user_index_), len(self.item_index_)

        mat = sps.coo_matrix((weights, (row_ind, col_ind)), shape=(n_rows, n_cols))

        return mat

    def _add_to_index(self, users_to_add=None, items_to_add=None) -> None:
        """[summary]

        Keyword Arguments:
            users_to_add {[type]} -- [description] (default: {None})
            items_to_add {[type]} -- [description] (default: {None})
        """
        if users_to_add:
            existing_user_index = self.user_index_
            self.user_index_ = pd.Index(np.unique(pd.concat(users_to_add)), name="user")
            if existing_user_index is not None:
                self.user_index_ = existing_user_index.append(
                    self.user_index_
                ).drop_duplicates(keep="first")
        if items_to_add:
            existing_item_index = self.item_index_
            self.item_index_ = pd.Index(np.unique(pd.concat(items_to_add)), name="item")
            if existing_item_index is not None:
                self.item_index_ = existing_item_index.append(
                    self.item_index_
                ).drop_duplicates(keep="first")

    def predict_for_user(
        self, user, items, user_features=None, item_features=None, **kwargs
    ):
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

        # TODO: Support user/item weights over features at prediction time

        good = iidx >= 0  # Limit to items in model
        items = np.array(items)
        good_items = items[good]
        good_iidx = iidx[good]

        scores = self.delegate.predict(uidx, good_iidx, **kwargs)
        res = pd.Series(scores, index=good_items)
        res = res.reindex(items)
        return res

    def _build_features(self, idx, idx_col, features, identity_features, normalize):
        n_rows = len(idx)
        feat_idx = pd.Index(np.unique(features.feature), name="feature")
        n_features = len(feat_idx)

        row_ind = idx.get_indexer(features[idx_col])

        if np.any(row_ind < 0):
            raise ValueError("Entry in features dataframe missing in ratings")
        col_ind = feat_idx.get_indexer(features.feature).astype(np.intc)

        if "value" in features.columns:
            data = np.require(features["value"], np.float32)
        else:
            data = np.ones_like(col_ind, dtype=np.float32)

        mat = self._build_sparse_features(
            data, row_ind, col_ind, n_rows, n_features, identity_features, normalize
        )
        return mat, n_features

    def _build_sparse_features(
        self, data, row_ind, col_ind, n_rows, n_features, identity_features, normalize
    ):

        mat = sps.csr_matrix((data, (row_ind, col_ind)), shape=(n_rows, n_features))

        if identity_features:
            mat = sps.hstack([mat, sps.eye(n_rows, format="csr", dtype=np.float32)])

        if normalize:
            if np.any(mat.getnnz(1) == 0):
                raise ValueError("Cannot normalize: Some rows have zero norm")
            from sklearn.preprocessing import normalize

            mat = normalize(mat, norm="l1", copy=False)

        return mat
