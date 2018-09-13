import time
import logging

from collections import namedtuple, deque

import numpy as np
import pandas as pd

from scipy.sparse import lil_matrix, hstack

from prune import *

logging.basicConfig(level=logging.INFO,
    format="%(levelname)s - %(asctime)s - %(msg)s",
    datefmt="%Y-%m-%d %H:%M:%S",
    handlers=[
        logging.FileHandler("log_knapsack.log"), # For debug
        logging.StreamHandler(),
    ])

def prepare_items(items=None, by=None):
    """ Convert list of namedtuples into dataframe and sort it """
    if isinstance(items, pd.DataFrame):
        items["density"] = items.eval("value / weight")
        if not by:
            by = "density" if items["density"].std() > 0.2 else "value"
            else_by = "density" if by != "density" else "value"
            by = [by, else_by]
        items.sort_values(by, ascending=False, inplace=True)
        logging.info("Sorted by {}".format(by))
        items["take"] = np.nan
        logging.info("First 5 items:\n{}".format(items.head()))
        return items
    return pd.DataFrame(columns=["value", "weight", "density", "take"])

def forward_step(domain, item, order, prev_state):
    """ Make DP forward step """
    logging.info("Item:\n{}".format(item))
    weight, value = int(item["weight"]), item["value"]
    lower_weight, upper_weight = \
        map(int, item[["lower_weight", "upper_weight"]].tolist())

    state = lil_matrix(prev_state, copy=True)

    if order == 0:
        state[0, weight] = value
        return True, state

    values, indeces = prev_state.data[0], prev_state.rows[0]
    lowest_index = max([w for w in indeces
                       if w <= lower_weight - weight] or [0])

    compare = deque()
    compare_col = lowest_index + weight
    compare_val = prev_state[0, lowest_index] + value
    compare_col = max(lower_weight, 0 + weight)

    changed = False
    cur_max = 0
    for col, val in zip(indeces, values):
        if lower_weight < col + weight <= upper_weight:
            compare.append((col + weight, val + value))
        if lower_weight < col <= upper_weight:
            # Go through all items in compare less than col
            while compare_col < col:
                # If compare_val > cur_max insert item, update cur_max
                # skip it in other case
                if compare_val > cur_max:
                    changed = True
                    state[0, compare_col] = cur_max = compare_val
                # Take next item from queue in any case
                try:
                    compare_col, compare_val = compare.popleft()
                except IndexError as e:
                    break
            # If compare_col == col and compare_val > val, insert copmare_val
            # and update cur_max
            if compare_col == col and cur_max < compare_val > val:
                changed = True
                state[0, compare_col] = cur_max = compare_val
            elif val > cur_max:
                cur_max = val
    # Fill rest
    while True:
        if compare_val > cur_max:
            state[0, compare_col] = cur_max = compare_val
            changed = True
        try:
            compare_col, compare_val = compare.popleft()
        except IndexError as e:
            break
    logging.info("Changed: {}\n".format(changed))
    state[:, :lower_weight] = 0
    state[:, upper_weight+1:] = 0
    return changed, state

def get_result(items):
    """ Return sum of values of taken items """
    return items.loc[items["take"] == 1, "value"].astype(int).sum()

def backward_path(order, ix, grid, items, clean=False):
    """ Find path to item with (order, ix) """
    taken = np.zeros(items.shape[0])
    while ix > 0:
        order = np.argmax(grid.tocsr()[:order, ix])
        logging.info("Order of item to take: {}".format(order))
        item_id = (items["order"] == order)
        logging.info("Item id: {}".format(np.where(item_id)[0]))
        logging.info("Take item\n{}".format(items.loc[item_id].T))
        taken[np.where(item_id)[0]] = 1
        ix -= int(items.loc[item_id, "weight"])
        logging.info("Next ix: {}".format(ix))
        if clean:
            # Remove rows with order higher than current order
            grid = lil_matrix(grid[:order,:])
            logging.debug("Number of items in domain: {}\tdomain shape: {}"
                .format(grid.count_nonzero(), grid.shape))
    # Fill rest
    return taken


class Knapsack(object):
    def __init__(self, capacity=0, items=None):
        self.capacity = capacity
        self.items = prepare_items(items)

        n_items = items.shape[0] if isinstance(items, pd.DataFrame) else 0
        self.n_items = n_items
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

    def get_result(self):
        """ Return sum of values of taken items """
        return get_result(self.items)

    def get_answer(self):
        """ Return answer as sequence of zeros and ones """
        return self.items.sort_index()["take"].astype(int).tolist()

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
        prev_state = lil_matrix((1, self.capacity + 1))
        for cur_id, item in search_items.iterrows():
            if ~np.isnan(self.items.loc[cur_id, "take"]):
                continue
            self.items.loc[cur_id, "order"] = order
            self.calculate_boundaries(cur_id)
            logging.info("Forward. order: {}\titem:\n{}"
                .format(order, cur_id, self.items.loc[item.name]))

            changed, new_state = forward_step(self.grid,
                self.items.loc[cur_id], order, prev_state)

            if changed:
                self.grid[order, :] = prev_state = new_state
                order += 1
            else:
                self.items.loc[cur_id, "order"] = np.nan
                self.items.loc[cur_id, "take"] = 0
                self.grid[order, :] = 0
            logging.debug("Number of items in prev_state: {}"
                .format(prev_state.count_nonzero()))
            if order % 25 == 0:
                logging.debug("Number of unsolved items: {}".format(
                    self.items[["order", "take"]].isnull().all(1).sum()))
                logging.debug("Number of items in domain: {}"
                    .format(self.grid.count_nonzero()))
            self.prune()
            self.feasibility_check()
        self.grid = self.grid[:order, :]
        self.items.drop("prune", axis=1, inplace=True)

    def backward(self):
        """ Find answer """
        logging.info("grid shape: {}".format(self.grid.shape))

        # Max value
        ix = int(self.grid.tocsr().max(0).argmax())
        logging.info("Result ix: {}".format(ix))

        order = self.grid.shape[0]
        self.items["take"] = \
            backward_path(order, ix, self.grid, self.items, clean=True)
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
        items = pd.DataFrame(items_list)
        knapsack = cls(capacity, items)
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
    """ Solve one of tasks """
    import os.path as op
    import time

    path = op.join("data", select_file_in("data"))
    # path = "data/ks_4_0"

    knapsack = Knapsack().load(path)
    answer = knapsack.solve()
    return answer

if __name__=="__main__":
    main()
