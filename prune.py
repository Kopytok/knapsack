import logging

import numpy as np

from scipy.sparse import lil_matrix, vstack

def prune(knapsack):
    """ Main pruning function. Prune oboth items and domain """
    pruned = prune_items(knapsack)
    if pruned:
        pruned = prune_domain(knapsack)
        return True
    return False

""" Items pruning part """

def prune_items(knapsack):
    """ Find items that must or must not be taken """
    sequence = [
        prune_exceeded_capacity,
    ]
    pruned = False
    for func in sequence:
        pruned = pruned or func(knapsack)
        logging.debug(
            "Number of solved items after {}: {}/{}".format(
                func.__name__,
                (~knapsack.items["take"].isnull()).sum(),
                knapsack.items.shape[0]))
    return pruned

def prune_zero_values(knapsack):
    """" Don't take items with 0 value """
    ix = (knapsack.items["value"] == 0)
    knapsack.items.loc[ix, "take"] = 0
    if ix.sum() > 0:
        logging.debug("Number of items with 0 value: {}"
            .format(ix.sum()))
        return True
    return False

def prune_fill_rest(knapsack):
    """ Fill with 0 unsolved items.
        Used in the end of backward stage """
    knapsack.items["take"].fillna(0, inplace=True)

def prune_exceeded_capacity(knapsack):
    """ Don't take items with weight > capacity """
    ix = (knapsack.items["weight"] > knapsack.capacity) & \
          knapsack.items["take"].isnull()
    knapsack.items.loc[ix, "take"] = 0
    if ix.sum() > 0:
        logging.debug("Number of items with too big weght: {}"
            .format(ix.sum()))
        return True
    return False

""" Domain pruning part """

def prune_domain(knapsack):
    """ Remove rows and columns from domain """
    sequence = [
        prune_remove_not_taken,
        prune_remove_taken,
    ]
    pruned = False
    for func in sequence:
        pruned = pruned or func(knapsack)
        logging.debug("Domain shape after {}: {}".format(
            func.__name__,
            knapsack.grid.shape))
    return pruned

def prune_clean_backward(knapsack, order):
    """ Remove from domain rows, observed after order """
    knapsack.grid = lil_matrix(knapsack.grid[:order,:])
    logging.debug("Number of items in domain: {}\tdomain shape: {}"
        .format(knapsack.grid.count_nonzero(), knapsack.grid.shape))

def prune_remove_not_taken(knapsack):
    """ Remove from domain rows for items with "take" == 0 """
    prune_incomming_not_taken(knapsack)
    prune_observed_not_taken(knapsack)

def prune_incomming_not_taken(knapsack):
    """ Decrease number of rows in domain by number of not taken items """
    # TODO Analyze
    pass
    # ix = (knapsack.items[["take", "prune"]] == 0).all(1)
    # subtrahend = ix.sum()
    # if subtrahend:
    #     knapsack.items.loc[ix, "prune"] = 1
    #     knapsack.grid = lil_matrix(knapsack.grid[:-subtrahend, :])
    #     logging.debug("Decreased domain n_rows by {}".format(subtrahend))

def prune_observed_not_taken(knapsack):
    """ Remove from domain rows for observed not taken items """
    pass

def prune_remove_taken(knapsack):
    """ Remove from domain rows for items with "take" == 1 and
        decrease capacity by their weight """
    pass
