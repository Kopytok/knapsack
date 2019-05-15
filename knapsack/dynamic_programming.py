from .imports import *


class DPForward(object):

    def __init__(self, capacity):
        self.capacity = capacity

        self.state = lil_matrix((1, capacity + 1))
        self.paths = dict()

    def reset_tmp(self):
        """ Copy state and paths """
        self.tmp_state = lil_matrix(self.state, copy=True)
        self.tmp_paths = self.paths.copy()

    def init_cmpr(self, lowest_index):
        """ Init cmpr values """
        self.cmpr_q = deque()
        self.cmpr_v = self.state[0, lowest_index] + self.value
        self.cmpr_w = max(self.l_weight, 0 + self.weight)
        self.cmpr_path = self.paths.get(lowest_index, set()) | {self.item_id}
        self.cur_max = 0

    def read_item(self, item):
        """ Save item values """
        self.weight, self.l_weight = \
            map(int, item[["weight", "lower_weight"]].tolist())
        self.value, self.item_id = item["value"], int(item.name)

    def take(self):
        """ Set cmpr_v, cmpr_path to tmp """
        self.tmp_state[0, self.cmpr_w] = self.cur_max = self.cmpr_v
        self.tmp_paths[self.cmpr_w] = self.cmpr_path

    def next(self):
        """ Next compare values """
        self.cmpr_w, self.cmpr_v = self.cmpr_q.popleft()
        self.cmpr_path = \
            self.paths.get(self.cmpr_w - self.weight) | {self.item_id}

    def next_with_take(self):
        """ Take item if it increases cur_max and take next value """
        if self.cur_max < self.cmpr_v:
            self.take()
        try:
            self.next()
        except IndexError as e:
            return True

    def compare(self):
        """ Compare items """
        for w, v in zip(self.weights, self.values):
            # Accumulate possible substitutes
            if self.l_weight <= w + self.weight <= self.capacity:
                self.cmpr_q.append((w + self.weight, v + self.value))

            if self.l_weight <= w <= self.capacity:
                # Go through all items in cmpr_q less than w
                while self.cmpr_w < w:
                    if self.next_with_take():
                        break

                if self.cmpr_w == w and self.cur_max < self.cmpr_v > v:
                    self.take()
                elif v > self.cur_max:
                    self.cur_max = v
        self.fill_rest()

    def fill_rest(self):
        """ Add items bigger than any previous """
        while True:
            if self.next_with_take():
                return

    def update_state(self):
        """ Update `state` & `path` from tmp
            and ignore weights outise window """
        for v, w in zip(self.tmp_state.data[0], self.tmp_state.rows[0]):
            if self.l_weight <= w <= self.capacity:
                self.state[0, w] = v
                self.paths[w] = self.tmp_paths[w]
            elif w in self.paths:
                self.state[0, w] = 0
                self.paths.pop(w)

    def forward_step(self, item):
        """ Perform dynamic programming forward step with item """
        self.read_item(item)

        # First item
        if len(self.paths) == 0:
            self.state[0, self.weight] = self.value
            self.paths[self.weight] = {self.item_id}
            self.reset_tmp()
            return None

        self.values, self.weights = self.state.data[0], self.state.rows[0]

        lowest_index = min([w for w in self.weights
                           if w <= self.l_weight - self.weight] or [0])
        self.init_cmpr(lowest_index)
        self.compare()
        self.update_state()


if __name__ == "__main__":
    pass
