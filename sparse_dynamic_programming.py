import time
import logging

from collections import namedtuple

import numpy as np
import pandas as pd

from scipy.sparse import lil_matrix

from prune import *

logging.basicConfig(level=logging.DEBUG,
    format="%(levelname)s - %(asctime)s - %(msg)s",
    datefmt="%Y-%m-%d %H:%M:%S",
    handlers=[
        logging.FileHandler("log_knapsack.log"), # For debug
        logging.StreamHandler(),
    ])

def read_item(line):
    """ Read one knapsack item """
    value, weight = map(int, line.split())
    return Item(value, weight, value / weight)

Item = namedtuple("Item", ["value", "weight", "density"])

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

    def answer(self):
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

    def get_row(self, order, lower_weight, upper_weight):
        """ Convert row from sparse into np.array """
        logging.debug("get_row. order: {}, lower_weight: {}, upper_weight: {}"
            .format(order, lower_weight, upper_weight))
        if order > -1:
            row = self.grid.tocsr()[order, lower_weight:upper_weight+1]\
                .toarray()
            row[0, 0] = self.grid.tocsr()[order, :lower_weight+1].max()
            return np.maximum.accumulate(row, axis=1)[0]
        return np.zeros(upper_weight - lower_weight + 1)

    def set_row(self, ix, row, floor):
        """ Add new row to the grid bottom """
        mask = np.hstack([[False], row[1:] != row[:-1]])
        lower_weight = np.argwhere(mask).min()
        upper_weight = np.argwhere(mask).max()
        logging.debug("set_row lower_weight: {}\tupper_weight: {}"
            .format(lower_weight, upper_weight))
        new_state = np.where(
            mask[lower_weight:upper_weight+1],
            row[lower_weight:upper_weight+1],
            0)
        self.grid[ix, floor+lower_weight:floor+upper_weight+1] = new_state

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

    def add_item(self, order, item):
        """ Evaluate item and expand grid """
        weight, value = item[["weight", "value"]].astype(int).tolist()
        lower_weight, upper_weight = self.items.loc[
            item.name,
            ["lower_weight", "upper_weight"]].astype(int).tolist()

        threshold = max(0, lower_weight - weight)
        logging.debug("add_item threshold: {}".format(threshold))
        state = self.get_row(order - 1, threshold, upper_weight)
        if_add = np.hstack([state[:weight], (state + value)[:-weight]])
        new_state = np.max([state, if_add], axis=0)

        self.set_row(order, new_state, threshold)
        logging.debug("items:\n{}".format(self.items.T))
        logging.debug("domain:\n{}".format(self.grid.todense()))
        if (new_state != state).all():
            self.items.loc[item.name, "take"] = 1
            logging.info("Filled 1 for item #{} (All changed)".format(order))
            return True
        elif (new_state == state).all():
            logging.info("Filled 0 for item #{} (No change)".format(order))
            self.items.loc[order, "take"] = 0
            prune_clean_one(self, order)
            return False
        return True

    def prune(self):
        """ Use prune as method """
        return prune(self)

    def forward(self):
        """ Fill domain """
        self.prepare_items_for_dp()
        self.prune()
        search_items = self.items.loc[self.items["take"].isnull()]

        order = 0
        for cur_id, item in search_items.iterrows():
            if ~np.isnan(self.items.loc[cur_id, "take"]):
                continue
            self.items.loc[cur_id, "order"] = order
            self.calculate_boundaries(cur_id)
            logging.debug("Forward. cur_id: {}\torder: {}\titem:\n{}"
                .format(cur_id, order, self.items.loc[item.name]))
            order += int(self.add_item(order, item))
            logging.debug("Number of items in domain: {}"
                .format(self.grid.count_nonzero()))
            self.prune()
            self.feasibility_check()
        self.items.drop("prune", axis=1, inplace=True)

    def backward(self):
        """ Find answer using filled domain """
        ix = int(self.grid.tocsr()[-1, :].argmax())
        logging.debug("Result ix: {}".format(ix))

        while ix > 0:
            col = self.grid.tocsr()[:, :ix+1].max(1)
            order = np.argmax(col)
            item_id = self.items["order"] == order
            logging.debug("Take item with order {}".format(order))
            self.items.loc[item_id, "take"] = 1
            ix -= int(self.items.loc[item_id, "weight"])
            logging.debug("Next ix: {}".format(ix))
            prune_clean_backward(self, order)

        prune_fill_rest(self)
        self.result = self.calculate_taken("value")
        logging.debug("Final items:\n{}".format(self.items.T))
        return self.answer()

    def solve(self):
        """ Run dynamic programming solver """
        t0 = time.time()
        logging.info("Filling domain")
        self.forward()
        logging.info("Finished forward")

        answer = self.backward()
        logging.info("Finished backward. Total time (sec): {}"
            .format(time.time() - t0))

        self.feasibility_check()
        logging.info("Resulting value: {}".format(self.result))
        logging.info("Selected items: {}".format(answer))
        return answer

    @classmethod
    def load(cls, path):
        """ Load knapsack from file """
        with open(path, "r") as f:
            items = f.readlines()

        n_items, capacity = map(int, items[0].split())
        logging.info("New data: {} items, {} capacity"
            .format(n_items, capacity))
        items_list = list()
        for item in items[1:]:
            if item.strip():
                row = read_item(item)
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
    # path = "data/ks_19_0"

    knapsack = Knapsack().load(path)
    answer = knapsack.solve()
    return answer

if __name__=="__main__":
    main()
