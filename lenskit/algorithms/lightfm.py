import logging
from itertools import starmap

import numpy as np
import pandas as pd
import scipy.sparse as sps

from lenskit.algorithms import Predictor

from ..matrix import sparse_ratings

_logger = logging.getLogger(__name__)


class LightFM(Predictor):
    """Lenskit interface to :py:mod:`LightFM.lightfm` that handles transformations between Lenskit
    ratings dataframes and matrix indexes. Additional arguments are passed through to LightFM.
    Separate interactions and weight matrices are possible by including a weight column.
    in the ratings dataframe.

    Keyword Arguments:
        user_identity_features {bool} -- Supplement user feature matrices with identity
            features. This has no effect if a user feature matrix is not supplied to `fit`.
            (default: {True})
        item_identity_features {bool} -- Supplement item feature matrices with identity
            features. This has no effect if an item feature matrix is not supplied to `fit`.
            (default: {True})
        *args: positional arguments passed to :py:class:`lightfm.LightFM` constructor
        **kwargs: keyword arguments passed to :py:class:`lightfm.LightFM` constructor.

    Attributes:
        delegate(lightfm.LightFM): The :py:mod:`lightfm.LightFM` delegate.
        _dataset(lightfm.LFMDataset): Builds interaction and feature matrices from dataframes
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
        """Fit the delegate LightFM model with ratings data. A separate `sample_weight_col` can be
        used to pass different interactions and weight matrices to LightFM, with the interactions
        drawn from the `rating` column.

        Arguments:
            ratings {[pandas.DataFrame]} -- The ratings data, with `user` and `item` columns, with
            an optional `rating` column.

        Keyword Arguments:
            sample_weight_col {[str]} -- Name of the `ratings` column to use for sample weights
                (default: {None})
            user_features {[pandas.DataFrame]} -- Dataframe with `user`, `feature`, and (optional)
                `value` to generate a user feature matrix. (default: {None})
            item_features {[pandas.DataFrame]} -- Dataframe with `item`, `feature`, and (optional)
                `value` to generate an item feature matrix. (default: {None})
            normalize_user_features {bool} -- Normalize user features by user (default: {True})
            normalize_item_features {bool} -- Normalize item features by item (default: {True})
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
            sample_weight_col {[type]} -- [description] (default: {None})
            user_features {[type]} -- [description] (default: {None})
            item_features {[type]} -- [description] (default: {None})
            normalize_user_features {bool} -- [description] (default: {True})
            normalize_item_features {bool} -- [description] (default: {True})

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
            user_features=user_features,
            item_features=item_features,
            sample_weight=sample_weight_mat,
            **kwargs
        )

        return self

    def _reset_state(self):
        self.delegate._reset_state()
        self._dataset._reset_state()

    def predict_for_user(self, user, items, user_features=None, item_features=None):
        """[summary]

        Arguments:
            user {[type]} -- [description]
            items {[type]} -- [description]

        Keyword Arguments:
            user_features {[type]} -- [description] (default: {None})
            item_features {[type]} -- [description] (default: {None})

        Returns:
            [type] -- [description]
        """
        df = pd.DataFrame({'user': user, 'item': items})
        return self.predict(df, user_features, item_features).set_index('item').score

    def predict(
        self,
        ratings,
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
            normalize_user_features {bool} -- [description] (default: {True})
            normalize_item_features {bool} -- [description] (default: {True})

        Returns:
            [type] -- [description]
        """
        self._dataset.partial_fit(ratings.user, ratings.item, user_features, item_features)

        users = self.dataset_.user_idx_.get_indexer(ratings.user, dtype=np.int32)
        items = self.dataset_.item_idx_.get_indexer(ratings.item, dtype=np.int32)

        ufeat_mat = self._dataset.build_user_features(user_features, normalize_user_features)
        ifeat_mat = self._dataset.build_item_features(item_features, normalize_item_features)

        scores = self.delegate.predict(
            users, items, user_features=ufeat_mat, item_features=ifeat_mat, **kwargs
            )
        return pd.DataFrame({'user': ratings.user, 'item': ratings.item, 'score': scores})


class LFMDataset:
    """Maintains mappings between user ids, item ids, and feature ids and matrix indices and generates sparse matrices for use by LightFM.
    """

    def __init__(self, user_identity_features, item_identity_features):
        """[summary]

        Arguments:
            user_identity_features {[type]} -- [description]
            item_identity_features {[type]} -- [description]
        """
        self.user_identity_features_ = user_identity_features
        self.item_identity_features_ = item_identity_features
        self._reset_state()

    def fit(self, users, items, user_features=None, item_features=None):
        """[summary]

        Arguments:
            users {[type]} -- [description]
            items {[type]} -- [description]

        Keyword Arguments:
            user_features {[type]} -- [description] (default: {None})
            item_features {[type]} -- [description] (default: {None})
        """
        self._reset_state()
        self.partial_fit(users, items, user_features, item_features)

    def partial_fit(
        self, users=None, items=None, user_features=None, item_features=None
    ):
        """[summary]

        Keyword Arguments:
            users {[type]} -- [description] (default: {None})
            items {[type]} -- [description] (default: {None})
            user_features {[type]} -- [description] (default: {None})
            item_features {[type]} -- [description] (default: {None})

        Returns:
            [type] -- [description]
        """
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

        self.add_to_idx_(user_idx, item_idx, ufeat_idx, ifeat_idx)

        # We can only add identity features once
        if self.user_identity_features_ and self.ufeat_identity_idx is None:
            self.ufeat_identity_idx_ = self.user_idx_
        if self.item_identity_features_ and self.ifeat_identity_idx_ is None:
            self.ifeat_identity_idx_ = self.item_idx_

        return self

    def add_to_idx_(self, user_idx=None, item_idx=None, ufeat_idx=None, ifeat_idx=None):
        """[summary]

        Keyword Arguments:
            user_idx {[type]} -- [description] (default: {None})
            item_idx {[type]} -- [description] (default: {None})
            ufeat_idx {[type]} -- [description] (default: {None})
            ifeat_idx {[type]} -- [description] (default: {None})
        """
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
        """[summary] Does not validate that users and items are mapped.

        Arguments:
            ratings {[type]} -- [description]

        Returns:
            [type] -- [description]
        """
        ratings_mat, self.user_idx_, self.item_idx_ = sparse_ratings(
            ratings, scipy="coo", users=self.user_idx_, items=self.item_idx_
        )
        return ratings_mat

    def build_weights(self, ratings, weight_col):
        """[summary]

        Arguments:
            ratings {[type]} -- [description]
            weight_col {[type]} -- [description]

        Raises:
            ValueError: [description]
            ValueError: [description]
            ValueError: [description]

        Returns:
            [type] -- [description]
        """
        row_ind = self.user_idx_.get_indexer(ratings.user).astype(np.intc)
        if np.any(row_ind < 0):
            raise ValueError("Users not in mapping. Call fit or partial_fit first.")
        col_ind = self.item_idx_.get_indexer(ratings.item).astype(np.intc)
        if np.any(col_ind < 0):
            raise ValueError("Items not in mapping. Call fit or partial_fit first.")

        if weight_col not in ratings.columns:
            raise ValueError("Weight column '%s' not in ratings", weight_col)
        else:
            vals = np.require(ratings[weight_col].values, np.float32)

        weights_mat = sps.coo_matrix((vals, (row_ind, col_ind)), shape=self.shape)

        return weights_mat

    @property
    def shape(self):
        return self.get_shape_(self.user_idx_, self.item_idx_)

    @property
    def user_features_shape(self):
        return self.get_shape_(self.user_idx_, self.ufeat_idx_)

    @property
    def item_features_shape(self):
        return self.get_shape_(self.item_idx_, self.ifeat_idx_)

    @staticmethod
    def get_shape_(rows, cols):
        n = len(rows) if rows else 0
        m = len(cols) if cols else 0
        return (n, m)

    def build_user_features(self, user_features, normalize):
        """[summary]

        Arguments:
            user_features {[type]} -- [description]
            normalize {[type]} -- [description]

        Raises:
            ValueError: [description]
            ValueError: [description]
            ValueError: [description]

        Returns:
            [type] -- [description]
        """
        if not {"user", "feature"} <= set(user_features.columns):
            raise ValueError(
                "Features must be dataframe with 'user' and 'feature' columns"
            )
        else:
            row_ind = self.user_idx_.get_indexer(user_features.feature).astype(np.intc)
            col_ind = self.ufeat_idx_.get_indexer(user_features.feature).astype(np.intc)

            if np.any(row_ind < 0):
                raise ValueError("Users in features not in dataset. Call fit or partial_fit first.")
            if np.any(col_ind < 0):
                raise ValueError("Features in user features not dataset. Call fit or partial_fit "
                                 "first.")

            shape = self.user_features_shape

            if "value" in user_features.columns:
                data = np.require(user_features.value, dtype=np.float32)
            else:
                data = np.ones_like(col_ind, dtype=np.float32)

            identity_idx = self.ufeat_identity_idx_

            return self._build_features_mat_(
                data, row_ind, col_ind, shape, normalize, identity_idx
            )

    def build_item_features(self, item_features, normalize):
        """[summary]

        Arguments:
            item_features {[type]} -- [description]
            normalize {[type]} -- [description]

        Raises:
            ValueError: [description]
            ValueError: [description]
            ValueError: [description]

        Returns:
            [type] -- [description]
        """
        if not {"item", "feature"} <= set(item_features.columns):
            raise ValueError(
                "Features must be dataframe with 'item' and 'feature' columns"
            )
        else:
            row_ind = self.item_idx_.get_indexer(item_features.feature).astype(np.intc)
            col_ind = self.ufeat_idx_.get_indexer(item_features.feature).astype(np.intc)

            if np.any(row_ind < 0):
                raise ValueError("items in features not in dataset")
            if np.any(col_ind < 0):
                raise ValueError("features in item features not dataset")

            shape = self.item_features_shape

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
                raise ValueError("Cannot normalize: Some rows have zero norm.")
            from sklearn.preprocessing import normalize

            mat = normalize(mat, norm="l1", copy=False)

        return mat

    def mapping(self):
        """[summary]

        Returns:
            [type] -- [description]
        """
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
        return pd.Series(np.arange(len(idx)).to_dict(), index=idx, name=name)
