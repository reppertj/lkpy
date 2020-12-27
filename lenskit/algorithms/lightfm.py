import logging

import numpy as np
import pandas as pd
import scipy

from ..matrix import sparse_ratings
from .mf_common import MFPredictor

_logger = logging.getLogger(__name__)


class LightFM(MFPredictor):
    """Lenskit interface to :py:mod:`LightFM.lightfm`

    Arguments:


    Attributes:
        delegate(lightfm.LightFM):
            The :py:mod:`lightfm` delegate model.
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

    def fit(self, ratings, **kwargs):
        """[summary]

        Arguments:
            ratings {[type]} -- [description]
        """
        if isinstance(ratings, pd.DataFrame):
            ratings, users, items = sparse_ratings(ratings, scipy=True)
            ratings = ratings.tocoo()
        elif isinstance(ratings, scipy.sparse.coo_matrix):
            users = pd.Index(np.unique(ratings.row), name="user")
            items = pd.Index(np.unique(ratings.col), name="item")
        else:
            raise TypeError(f"unsupported type: {type(ratings)}")

        _logger.info("fitting LightFM model")
        self.delegate.fit(ratings, **kwargs)

        self.matrix_ = ratings
        self.user_index_ = users
        self.item_index_ = items

        return self

    def predict_for_user(self, user, items, ratings):
        return None
