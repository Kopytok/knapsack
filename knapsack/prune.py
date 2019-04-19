from .imports import *

def prune(knapsack):
    """ Main pruning function """
    pruned = prune_paths(knapsack)
    if pruned:
        prune_items(knapsack)
        return True
    return False

""" Items pruning part """

def prune_items(knapsack):
    """ Find items that must or must not be taken """
    sequence = [
        prune_exceeded_free_space,
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

def prune_exceeded_free_space(knapsack):
    """ Don't take items with weight > free_space """
    free_space = knapsack.capacity - knapsack.filled_space
    ix = (knapsack.items["weight"] > free_space) & \
          knapsack.items["take"].isnull()
    knapsack.items.loc[ix, "take"] = 0
    if ix.sum() > 0:
        logging.debug("Number of items with too big weight: {}"
            .format(ix.sum()))
        return True
    return False

""" Domain pruning part """

def prune_paths(knapsack):
    """ Remove rows and columns from domain """
    sequence = [
        prune_remove_taken,
        prune_incomming_not_taken,
    ]
    pruned = False
    for func in sequence:
        pruned = pruned or func(knapsack)
    return pruned

def prune_incomming_not_taken(knapsack):
    """ Do not observe incomming items that do not fit """
    pass

def prune_remove_taken(knapsack):
    """ Find items that must be taken. Thus lower paths and state """
    must_take = reduce(lambda x, y: x & y, knapsack.paths.values())
    logging.debug("must_take: {}".format(must_take))
    if len(must_take):
        for item_id in must_take:
            logging.info("take obvious: {}".format(item_id))
            knapsack.take_item(item_id)
        return True
    return False
