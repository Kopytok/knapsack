import time
import logging
from copy import deepcopy
from collections import namedtuple

import numpy as np
import pandas as pd

from scipy.sparse import vstack, csr_matrix

logging.basicConfig(level=logging.DEBUG,
    format="%(levelname)s - %(asctime)s - %(msg)s",
    datefmt="%Y-%m-%d %H:%M:%S",
    handlers=[
        logging.FileHandler("text_log.log"), # For debug
        logging.StreamHandler(),
    ])

def line_to_numbers(line):
    """ Split line into 2 digits and convert them to int """
    return tuple(map(int, line.split()))

def read_item(line):
    """ Read one knapsack item """
    value, weight = line_to_numbers(line)
    return Item(value, weight, value / weight)

Item = namedtuple("Item", ["value", "weight", "density"])

def prepare_items(items=None):
    if items:
        df = pd.DataFrame(items)
        by = "density" if df["density"].std() > 0.2 else "value"
        else_by = "density" if by != "density" else "value"
        logging.info("Sorted by {}".format(by))
        df.sort_values([by, else_by], ascending=False, inplace=True)
        df["select"] = np.nan
        logging.info("First 5 items:\n{}".format(df.head()))
        return df
    else:
        return


class Domain(object):
    def __init__(self, n_items=0, capacity=0, items=None):
        self.n_items = n_items
        self.capacity = capacity
        self.items = prepare_items(items)

        self.numbers = np.linspace(0, capacity, capacity + 1)
        self.grid = csr_matrix((1, capacity + 1))
        self.cur_item_id = 0

        self.result = 0

    def __len__(self):
        return self.grid.shape[0] - 1

    def __repr__(self):
        return "Knapsack. Capacity: {}, items: {}"\
            .format(self.capacity, self.n_items)

    def get_row(self, row_ix):
        """ Convert row from sparse into np.array """
        row = self.grid[row_ix, :].toarray()
        return np.maximum.accumulate(row, axis=1)[0]

    def add_row(self, row):
        """ Add new row to the grid bottom """
        mask = np.hstack([[False], row[1:] != row[:-1]])
        new_state = np.where(mask, row, 0)
        self.grid = vstack([self.grid, new_state]).tocsr()

    def add_item(self, n, item):
        """ Evaluate item and expand grid """
        state = self.get_row(-1)

        weight = int(item.weight)
        if weight < self.capacity:
            item_value = np.where(self.numbers > weight, item.value, 0)
            if_add = np.hstack(
                [state[:weight], (state + item.value)[:-weight]])
            new_state = np.max([state, if_add], axis=0)
            self.add_row(new_state)
            if (new_state[:-weight] != state[:-weight]).all():
                logging.info("Filled 1 for {}".format(n))
                self.items.loc[n, "select"] = 1
        else:
            logging.info("Filled 0 for {}".format(n))
            self.items.loc[n, "select"] = 0

    def forward(self):
        """ Fill domain """
        for n, item in self.items.iterrows():
            self.cur_item_id = n
            if np.isnan(item["select"]):
                self.add_item(n, item)

    def backward(self):
        """ Find answer using filled domain """
        last_row = self.get_row(-1)
        ix = np.argmax(last_row) # First weight with max value
        self.result = int(np.max(last_row))

        for n, (i, item) in enumerate(self.items.iloc[::-1, :].iterrows()):
            weight = int(self.items.loc[i, "weight"])
            if np.isnan(item["select"]):
                cur = self.get_row(-n)
                prev = self.get_row(-(n - 1))
                if cur[ix] == prev[ix] == 0:
                    break
                logging.debug("cur[ix]: {}".format(cur[ix]))
                logging.debug("prev[ix]: {}".format(prev[ix]))
                self.items.loc[i, "select"] = (cur[ix] != prev[ix])

            select = self.items.loc[i, "select"]
            if select:
                logging.debug("Select item:\n{}".format(item))
            ix -= weight if select else 0
            logging.debug("ix: {}".format(ix))
            if ix == 0:
                break

        self.items["select"].fillna(0, inplace=True)
        return self.items.sort_index()["select"].astype(int).tolist()

    def solve(self):
        t0 = time.time()
        logging.info("Filling domain")
        self.forward()
        logging.info("Finished filling domain")
        answer = self.backward()
        logging.info("Finished in (sec): {}".format(time.time() - t0))
        logging.info("Resulting value: {}".format(self.result))
        logging.info("Selected items: {}".format(answer))
        return answer

    @classmethod
    def load(cls, path):
        with open(path, "r") as f:
            items = f.readlines()

        n_items, capacity = line_to_numbers(items[0])
        logging.info("New data: {} items, {} capacity".format(n_items, capacity))
        items_list = list()
        for item in items[1:]:
            row = read_item(item)
            items_list.append(row)
        d = cls(n_items, capacity, items_list)
        return d


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

    d = Domain().load(path)
    t0 = time.time()
    answer = d.solve()

if __name__=="__main__":
    main()
