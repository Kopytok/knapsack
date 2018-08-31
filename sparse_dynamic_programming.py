import time
import logging

from collections import namedtuple, deque

import numpy as np
import pandas as pd

from scipy.sparse import lil_matrix, hstack

from prune import *

logging.basicConfig(level=logging.DEBUG,
    format="%(levelname)s - %(asctime)s - %(msg)s",
    datefmt="%Y-%m-%d %H:%M:%S",
    handlers=[
        logging.FileHandler("log_knapsack.log"), # For debug
        logging.StreamHandler(),
    ])

def prepare_items(items=None, by=None):
    """ Convert list of namedtuples into dataframe and sort it """
    if items:
        df = pd.DataFrame(items)
        if not by:
            by = "density" if df["density"].std() > 0.2 else "value"
            else_by = "density" if by != "density" else "value"
            by = [by, else_by]
        df.sort_values(by, ascending=False, inplace=True)
        logging.info("Sorted by {}".format(by))
        df["take"] = np.nan
        logging.info("First 5 items:\n{}".format(df.head()))
        return df
    return pd.DataFrame(columns=["value", "weight", "density", "take"])

def forward_step(domain, item, order, state):
    """ Make DP forward step """
    logging.debug("Item:\n{}".format(item))
    weight, value = int(item["weight"]), item["value"]
    lower_weight, upper_weight = \
        map(int, item[["lower_weight", "upper_weight"]].tolist())
    if order == 0:
        domain[order, weight] = value
        order += 1
        state[0, weight] = value
        return True
    values, indeces = state.data[0], state.rows[0]
    first_index = max([w for w in indeces
                       if w <= lower_weight - weight] or [0])

    compare = deque()
    compare_col = first_index + weight
    compare_val = state[0, first_index] + value
    compare_col = max(lower_weight, 0 + weight)

    changed = False
    cur_max = 0
    for col, val in zip(indeces, values):
        if lower_weight - weight < col <= upper_weight - weight:
            compare.append((col + weight, val + value))
        if lower_weight <= col <= upper_weight:
            # Go through all items in compare less than col
            while compare_col < col:
                # If compare_val > cur_max insert item, update cur_max
                # skip it in other case
                if compare_val > cur_max:
                    changed = True
                    domain[order, compare_col] = state[0, compare_col]\
                        = cur_max = compare_val
                # Take next item from queue in any case
                try:
                    compare_col, compare_val = compare.popleft()
                except IndexError as e:
                    break
            # If compare_col == col and compare_val > val, insert copmare_val
            # and update cur_max
            if compare_col == col and cur_max < compare_val > val:
                changed = True
                domain[order, compare_col] = state[0, compare_col]\
                    = cur_max = compare_val
            elif val > cur_max:
                domain[order, col] = cur_max = val
    # Fill rest
    while True:
        if compare_val > cur_max:
            domain[order, compare_col] = state[0, compare_col]\
                = cur_max = compare_val
            changed = True
        try:
            compare_col, compare_val = compare.popleft()
        except IndexError as e:
            break
    logging.debug("Changed: {}\n".format(changed))
    state[:, :lower_weight] = 0
    state[:, upper_weight+1:] = 0
    return changed


class Knapsack(object):
    def __init__(self, n_items=0, capacity=0, items=None):
        self.n_items = n_items
        self.capacity = capacity
        self.items = prepare_items(items)
        prune_zero_values(self)

        self.numbers = np.linspace(0, capacity, capacity + 1)
        self.grid = lil_matrix((n_items, capacity + 1))
        self.result = 0

    def __repr__(self):
        return "Knapsack. Capacity: {}, items: {}"\
            .format(self.capacity, self.n_items)

    def feasibility_check(self):
        """ Check if total weight of taken items is
            less than knapsack capacity """
        assert self.items.loc[self.items["take"] == 1, "weight"].sum() \
            <= self.capacity, "Not feasible answer. Exceeded capacity."

    def get_answer(self):
        """ Return answer as sequence of zeros and ones """
        return self.items.sort_index()["take"].astype(int).tolist()

    def get_result(self):
        """ Return sum of values of taken items """
        return self.items.loc[self.items["take"] == 1, "value"].astype(int).sum()

    def eval_left(self, param="value", order=None):
        """ Return sum of param for untouched items """
        order = order or self.capacity
        ix = ((self.items["take"].isnull() & self.items["order"].isnull()) |
              (self.items["order"] > order))
        return self.items.loc[ix, param].sum()

    def prepare_items_for_dp(self):
        aux_columns = [
            "order",
            "avail_value",
            "upper_weight",
            "lower_weight",
        ]
        for col in aux_columns:
            self.items[col] = np.nan
        self.items["prune"] = 0

    def calculate_taken(self, value="value"):
        """ Calculate total of taken items values (value or weight) """
        return self.items.loc[self.items["take"] == 1, value].sum()

    def calculate_boundaries(self, ix):
        """ Calculate useful values for taken item """
        order = self.items.loc[ix, "order"]
        all_taken = self.items.loc[self.items["order"] <= order]
        self.items.loc[ix, "upper_weight"] = \
            min(all_taken["weight"].sum(), self.capacity)
        self.items.loc[ix, "avail_value"] = self.eval_left("value", order)
        self.items.loc[ix, "lower_weight"] = \
            max(0, self.capacity - self.eval_left("weight", order)
                - self.items.loc[ix, "weight"])

    def prune(self):
        """ Use prune as method """
        return prune(self)

    def forward(self):
        """ Fill domain """
        self.prepare_items_for_dp()
        self.prune()
        search_items = self.items.loc[self.items["take"].isnull()]

        order = 0
        state = lil_matrix((1, self.capacity + 1))
        for cur_id, item in search_items.iterrows():
            if ~np.isnan(self.items.loc[cur_id, "take"]):
                continue
            self.items.loc[cur_id, "order"] = order
            self.calculate_boundaries(cur_id)
            logging.debug("Forward. order: {}\titem:\n{}"
                .format(cur_id, order, self.items.loc[item.name]))
            changed = forward_step(self.grid, self.items.loc[cur_id], order,
                                   state)
            if changed:
                order += 1
            else:
                self.items.loc[cur_id, "order"] = np.nan
                self.items.loc[cur_id, "take"] = 0
            logging.debug("Number of items in state: {}"
                .format(state.count_nonzero()))
            logging.debug("Number of unsolved items: {}".format(
                self.items[["order", "take"]].isnull().all(1).sum()))
            logging.debug("Number of items in domain: {}"
                .format(self.grid.count_nonzero()))
            self.prune()
            self.feasibility_check()
        self.items.drop("prune", axis=1, inplace=True)

    def backward(self):
        """ Find answer using filled domain """
        max_order = order = self.items["order"].max() # Redundant. Should be last row.
        ix = int(self.grid.tocsr()[max_order, :].argmax())
        logging.debug("Result ix: {}".format(ix))

        while ix > 0 and order > 0: # Order should be redundant
            col = self.grid.tocsr()[:, :ix+1].max(1)
            order = np.argmax(col)
            item_id = self.items["order"] == order
            logging.debug("Take item with order {}".format(order))
            self.items.loc[item_id, "take"] = 1
            ix -= int(self.items.loc[item_id, "weight"])
            logging.debug("Next ix: {}".format(ix))
            prune_clean_backward(self, order)

        prune_fill_rest(self)
        logging.debug("Final items:\n{}".format(self.items.T))
        self.result = self.get_result()

    def solve(self):
        """ Run dynamic programming solver """
        t0 = time.time()
        logging.info("Filling domain")
        self.forward()
        logging.info("Finished forward")

        self.backward()
        logging.info("Finished backward. Total time (sec): {}"
            .format(time.time() - t0))

        self.feasibility_check()
        answer = self.get_answer()
        logging.info("Resulting value: {}".format(self.result))
        logging.info("Selected items: {}"
            .format(answer))
        return answer

    @classmethod
    def load(cls, path):
        """ Load knapsack from file """
        with open(path, "r") as f:
            items = f.readlines()

        n_items, capacity = map(int, items[0].split())
        logging.info("New data: {} items, {} capacity"
            .format(n_items, capacity))

        Item = namedtuple("Item", ["value", "weight", "density"])
        items_list = list()
        for item in items[1:]:
            if item.strip():
                value, weight = map(int, item.split())
                row = Item(value, weight, value / weight)
                items_list.append(row)
        knapsack = cls(n_items, capacity, items_list)
        return knapsack


def select_file_in(folder, rows=8):
    """ Select menu for input directory """
    import os

    files = [file for file in os.listdir(folder) if "ks" in file]
    page  = 0
    while True:
        for i, name in zip(range(rows), files[page * rows:(page + 1) * rows]):
            print(i, name)
        try:
            choice = int(input(
                "Select file. (8 for prev page, 9 for next page)\n"))
        except ValueError as e:
            continue
        if choice == 9 and len(files):
            page += 1
        elif choice == 8 and page > 0:
            page -= 1
        elif choice in list(range(rows)):
            try:
                return files[page * 8 + choice]
            except IndexError as e:
                continue

def main():
    import os.path as op
    import time

    path = op.join("data", select_file_in("data"))
    # path = "data/ks_4_0"

    knapsack = Knapsack().load(path)
    answer = knapsack.solve()
    return answer

if __name__=="__main__":
    main()
