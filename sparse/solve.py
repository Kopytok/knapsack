import logging
from copy import deepcopy

import numpy as np
from scipy.sparse import vstack, csr_matrix

from knapsack_queue import Queue

logging.basicConfig(level=logging.INFO,
    format="%(levelname)s - %(asctime)s - %(msg)s",
    datefmt="%Y-%m-%d %H:%M:%S")


class Domain(object):
    def __init__(self, queue):
        self.n_items = queue.n_items
        self.capacity = queue.capacity
        self.items = list(queue._queue)
        self.numbers = np.linspace(0, queue.capacity, queue.capacity + 1)
        self.grid = csr_matrix((1, queue.capacity + 1))
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
        item_value = np.where(self.numbers > item.weight, item.value, 0)
        if item.weight < self.capacity:
            shifted_state = np.hstack([[0] * item.weight, (state + item.value)[:-item.weight]])
            new_state = np.max([state, shifted_state], axis=0)
        else:
            new_state = state

        # Add row to domain
        self.add_row(new_state)

    def forward(self):
        """ Fill domain """
        for item in self.items:
            self.add_item(item)

    def backward(self):
        """ Find answer using filled domain """
        prev = self.get_row(-1)

        ix = np.argmax(prev)
        self.result = np.max(prev)

        answer = dict()
        for i in range(self.n_items-1, -1, -1):
            cur = self.get_row(i)
            item = self.items[i]
            if cur[ix] == prev[ix]:
                answer[item.id] = 0
            else:
                answer[item.id] = 1
                ix -= item.weight
            prev = cur
        return [value for (key, value) in sorted(answer.items())]

    def solve(self):
        self.forward()
        logging.info("Finished filling domain")
        return self.backward()


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

    data_folder = op.join("..", "data")
    filename = select_file(data_folder)
    path = op.join(data_folder, filename)

    q = Queue.read_queue(path)
    q.sort("density", descending=True)
    logging.info(q)

    d = Domain(q)
    t0 = time.time()
    answer = d.solve()
    logging.info("Finished in (sec): {}".format(time.time() - t0))
    logging.info("Resulting value: {}".format(d.result))
    logging.info("Selected items: {}".format(answer))

if __name__=="__main__":
    main()
