import logging
from copy import deepcopy
from collections import namedtuple

import numpy as np
import pandas as pd

from scipy.sparse import vstack, csr_matrix

logging.basicConfig(level=logging.INFO,
    format="%(levelname)s - %(asctime)s - %(msg)s",
    datefmt="%Y-%m-%d %H:%M:%S")

def line_to_numbers(line):
    """ Split line into 2 digits and convert them to int """
    return tuple(int(num) for num in line.split())

def read_item(line):
    """ Read one knapsack item """
    value, weight = line_to_numbers(line)
    return Item(value, weight, value / weight)

Item = namedtuple("Item", ["value", "weight", "density"])


class Domain(object):
    def __init__(self, n_items=0, capacity=0, items=None):
        self.n_items = n_items
        self.capacity = capacity
        self.items = pd.DataFrame(items)

        self.numbers = np.linspace(0, capacity, capacity + 1)
        self.grid = csr_matrix((1, capacity + 1))
        self.result = 0

    def __len__(self):
        return self.grid.shape[0] - 1

    def __repr__(self):
        return "Knapsack. Capacity: {}, items: checked {}\{}"\
            .format(self.capacity, len(self), self.n_items)

    def get_row(self, row_ix):
        """ Convert row from sparse into np.array """
        row = self.grid[row_ix, :].toarray()
        return np.maximum.accumulate(row, axis=1)[0]

    def add_row(self, row):
        """ Add new row to the grid bottom """
        mask = np.hstack([[False], row[1:] != row[:-1]])
        new_state = np.where(mask, row, 0)
        self.grid = vstack([self.grid, new_state]).tocsr()

    def add_item(self, item):
        """ Evaluate item and expand grid """
        state = self.get_row(-1)

        # Evaluate
        try:
            item_value = np.where(self.numbers > item.weight, item.value, 0)
        except AttributeError as e:

            logging.info("Item: {}".format(item))
            raise e
        if item.weight < self.capacity:
            weight = int(item.weight)
            shifted_state = np.hstack([[0] * weight, (state + item.value)[:-weight]])
            new_state = np.max([state, shifted_state], axis=0)
        else:
            new_state = state

        # Add row to domain
        self.add_row(new_state)

    def forward(self):
        """ Fill domain """
        for n, item in self.items.iterrows():
            self.add_item(item)

    def backward(self):
        """ Find answer using filled domain """
        prev = self.get_row(-1)
        ix = np.argmax(prev)
        self.result = np.max(prev)

        answer = dict()
        for i in range(self.n_items-1, -1, -1):
            cur = self.get_row(i)
            item = self.items.iloc[i]
            if cur[ix] == prev[ix]:
                answer[item.name] = 0
            else:
                answer[item.name] = 1
                ix -= int(item.weight)
            prev = cur
        return [value for (key, value) in sorted(answer.items())]

    def solve(self):
        self.forward()
        logging.info("Finished filling domain")
        return self.backward()

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

    # q = Queue.read_queue(path)
    # q.sort("density", descending=True)
    # logging.info(q)

    d = Domain().load(path)
    t0 = time.time()
    answer = d.solve()
    logging.info("Finished in (sec): {}".format(time.time() - t0))
    logging.info("Resulting value: {}".format(d.result))
    logging.info("Selected items: {}".format(answer))

if __name__=="__main__":
    main()
