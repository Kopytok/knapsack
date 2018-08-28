def prune(knapsack):
    pass

def prune_zero_values(knapsack):
    """" Don't take items with 0 value """
    ix = (knapsack.items["value"] == 0)
    knapsack.items.loc[ix, "take"] = 0

def prune_exceeded_capacity(knapsack):
    """ Don't take items with weight > capacity """
    ix = (knapsack.items["weight"] > knapsack.capacity)
    knapsack.items.loc[ix, "take"] = 0
