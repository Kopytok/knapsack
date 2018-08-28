import time
import logging

from collections import namedtuple

import numpy as np
import pandas as pd

from scipy.sparse import vstack, lil_matrix

from prune import *

logging.basicConfig(level=logging.INFO,
    format="%(levelname)s - %(asctime)s - %(msg)s",
    datefmt="%Y-%m-%d %H:%M:%S",
    handlers=[
        logging.FileHandler("log_knapsack.log"), # For debug
        logging.StreamHandler(),
    ])

def line_to_numbers(line):
    """ Split line into 2 digits and convert them to int """
    return map(int, line.split())

def read_item(line):
    """ Read one knapsack item """
    value, weight = line_to_numbers(line)
    return Item(value, weight, value / weight)

Item = namedtuple("Item", ["value", "weight", "density"])

def prepare_items(items=None, by=None):
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
    return None


class Knapsack(object):
    def __init__(self, n_items=0, capacity=0, items=None):
        self.n_items = n_items
        self.capacity = capacity
        self.items = prepare_items(items)

        self.numbers = np.linspace(0, capacity, capacity + 1)
        self.grid = lil_matrix((n_items, capacity + 1))

        self.result = 0

    def __len__(self):
        return self.grid.shape[0] - 1

    def __repr__(self):
        return "Knapsack. Capacity: {}, items: {}"\
            .format(self.capacity, self.n_items)

    def eval_left(self, col="value"):
        """ Return sum of col for untouched items """
        return self.items.loc[self.items["take"].isnull(), col].sum()

    def get_row(self, row_ix):
        """ Convert row from sparse into np.array """
        if row_ix > -1:
            row = self.grid.tocsr()[row_ix, :].toarray()
            return np.maximum.accumulate(row, axis=1)[0]
        else:
            return np.zeros(self.capacity + 1)

    def set_row(self, ix, row):
        """ Add new row to the grid bottom """
        mask = np.hstack([[False], row[1:] != row[:-1]])
        new_state = np.where(mask, row, 0)
        self.grid[ix, :] = new_state

    def add_item(self, cur_id, item):
        """ Evaluate item and expand grid """
        weight, value = int(item["weight"]), int(item["value"])

        state = self.get_row(cur_id - 1)
        if_add = np.hstack([state[:weight], (state + value)[:-weight]])
        new_state = np.max([state, if_add], axis=0)
        self.set_row(cur_id, new_state)
        logging.debug("domain:\n{}".format(self.grid.todense()))
        if (new_state[weight:] == state[weight:]).all():
            logging.info("Filled 0 for item #{} (No change)".format(cur_id))
            self.items.loc[cur_id, "take"] = 0

    def forward(self):
        """ Fill domain """
        self.items["order"] = np.nan
        prune_exceeded_capacity(self)

        search_items = self.items.loc[self.items["take"].isnull()]

        order = 0
        prev_id = -1
        for cur_id, item in search_items.iterrows():
            logging.debug("Forward. cur_id: {}\torder: {}\titem:\n{}"
                .format(cur_id, order, item))
            self.add_item(order, item)
            self.items.loc[cur_id, "order"] = order
            order += 1
            prev_id = cur_id
            prune(self)

    def backward(self):
        """ Find answer using filled domain """
        search_items = self.items.loc[~self.items["order"].isnull()]\
            .sort_values("order", ascending=False).copy()
        logging.debug("Backward search items:\n{}".format(search_items))

        last_item_id = search_items["order"].max()
        last = self.get_row(last_item_id)
        ix = np.argmax(last)
        logging.debug("Result ix: {}".format(ix))

        prev_id = -1
        for cur_id, item in search_items.iterrows():
            logging.debug("Backward. cur_id: {}\titem:\n{}"
                .format(cur_id, item))
            weight = int(item["weight"])

            if prev_id == -1:
                cur = self.get_row(item["order"])
            prev = self.get_row(item["order"] - 1)
            logging.debug("cur[ix]: {}".format(cur[ix]))
            logging.debug("prev[ix]: {}".format(prev[ix]))

            take = int(cur[ix] != prev[ix])
            self.items.loc[cur_id, "take"] = take
            logging.debug("Take" if take else "Leave")

            ix -= weight if take else 0
            logging.debug("ix: {}".format(ix))
            if ix == 0:
                break
            cur = prev

        # Since ix == 0, don't take rest items
        self.items["take"].fillna(0, inplace=True)
        # Calculate resulting value
        self.result = self.items.loc[self.items["take"] == 1, "value"].sum()
        logging.debug("Final items:\n{}".format(self.items))
        return self.items.sort_index()["take"].astype(int).tolist()

    def solve(self):
        """ Run dynamic programming solver """
        t0 = time.time()
        prune_zero_values(self)
        logging.info("Filling domain")
        self.forward()
        logging.info("Finished forward")
        answer = self.backward()
        logging.info("Finished backward. Total time (sec): {}"
            .format(time.time() - t0))
        logging.info("Resulting value: {}".format(self.result))
        logging.info("Selected items: {}".format(answer))
        return answer

    @classmethod
    def load(cls, path):
        """ Load knapsack from file """
        with open(path, "r") as f:
            items = f.readlines()

        n_items, capacity = line_to_numbers(items[0])
        logging.info("New data: {} items, {} capacity"
            .format(n_items, capacity))
        items_list = list()
        for item in items[1:]:
            if item.strip():
                row = read_item(item)
                items_list.append(row)
        knapsack = cls(n_items, capacity, items_list)
        return knapsack


def select_file(folder, rows=8):
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

    data_folder = "data"
    filename = select_file(data_folder)
    path = op.join(data_folder, filename)

    # path = "data/ks_4_0"

    knapsack = Knapsack().load(path)
    answer = knapsack.solve()
    return answer

if __name__=="__main__":
    main()
