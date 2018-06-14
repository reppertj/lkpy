"""
Data set cross-folding.
"""

from collections import namedtuple
import logging

import numpy as np

TTPair = namedtuple('TTPair', ['train', 'test'])

_logger = logging.getLogger(__package__)

def partition_rows(data, partitions):
    """
    Partition a frame of ratings or other datainto train-test partitions.  This function does not
    care what kind of data is in `data`, so long as it is a Pandas DataFrame (or equivalent).

    :param data: a data frame containing ratings or other data you wish to partition.
    :type data: `pd.DataFrame` or equivalent
    :param partitions: the number of partitions to produce
    :type partitions: integer
    :rtype: iterator
    :returns: an iterator of train-test pairs
    """
    _logger.info('partitioning %d ratings into %d partitions', len(data), partitions)

    # create an array of indexes
    rows = np.arange(len(data))
    # shuffle the indices & split into partitions
    np.random.shuffle(rows)
    test_sets = np.array_split(rows, partitions)

    # convert each partition into a split
    for i, ts in enumerate(test_sets):
        test = data.iloc[ts,:]
        trains = test_sets[:i] + test_sets[(i+1):]
        train_idx = np.concatenate(trains)
        train = data.iloc[train_idx,:]
        yield TTPair(train, test)

def sample_rows(data, partitions, size, disjoint=True):
    """
    Sample train-test a frame of ratings into train-test partitions.  This function does not care what kind
    of data is in `data`, so long as it is a Pandas DataFrame (or equivalent).

    :param data: a data frame containing ratings or other data you wish to partition.
    :type data: `pd.DataFrame` or equivalent
    :param partitions: the number of partitions to produce
    :type partitions: integer
    :rtype: iterator
    :returns: an iterator of train-test pairs
    """

    if disjoint and partitions * size >= len(data):
        _logger.warn('wanted %d disjoint splits of %d each, but only have %d rows; partitioning',
                     partitions, size, len(data))
        for p in partition_rows(data, partitions):
            yield p
        return

    # create an array of indexes
    rows = np.arange(len(data))

    if disjoint:
        _logger.info('creating %d disjoint samples of size %d', partitions, size)
        ips = _disjoint_sample(rows, partitions, size)        

    else:
        _logger.info('taking %d samples of size %d', partitions, size)
        ips = _n_samples(rows, partitions, size)
    
    for ip in ips:
        yield TTPair(data.iloc[ip.train,:], data.iloc[ip.test,:])

def _disjoint_sample(idxes, n, size):
    # shuffle the indices & split into partitions
    np.random.shuffle(idxes)

    # convert each partition into a split
    for i in range(n):
        start = i * size
        test = idxes[start:start+size]
        train = np.concatenate((idxes[:start], idxes[start+size:]))
        yield TTPair(train, test)

def _n_samples(idxes, n, size):
    for i in range(n):
        test = np.random.choice(idxes, size, False)
        train = np.setdiff1d(idxes, test, assume_unique=True)
        yield TTPair(train, test)

def partition_users(data, partitions, holdout):
    """
    Partition a frame of ratings or other data into train-test partitions user-by-user.
    This function does not care what kind of data is in `data`, so long as it is a Pandas DataFrame
    (or equivalent) and has a `user` column.

    :param data: a data frame containing ratings or other data you wish to partition.
    :type data: `pd.DataFrame` or equivalent
    :param partitions: the number of partitions to produce
    :type partitions: integer
    :param holdout: the number of test rows per user
    :type hodlout: integer
    :rtype: iterator
    :returns: an iterator of train-test pairs
    """

    user_col = data['user']
    users = user_col.unique()
    _logger.info('partitioning %d rows for %d users into %d partitions',
                 len(data), len(users), partitions)

    # create an array of indexes into user row
    rows = np.arange(len(users))
    # shuffle the indices & split into partitions
    np.random.shuffle(rows)
    test_sets = np.array_split(rows, partitions)

    # convert each partition into a split
    for i, ts in enumerate(test_sets):
        # get our users!
        test_us = users[ts]
        # sample the data frame
        test = data[data.user.isin(test_us)].groupby('user').apply(lambda udf: udf.sample(holdout))
        # get rid of the group index
        test = test.reset_index(0, drop=True)
        # now test is indexed on the data frame! so we can get the rest
        rest = data.index.difference(test.index)
        train = data.loc[rest]
        yield TTPair(train, test)
