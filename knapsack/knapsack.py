from .imports import *
from .prune import *
from .open_data import *

def forward_step(item, state, paths):
    """ Make DP forward step """
    logging.info("Item:\n{}".format(item))
    weight, lower_weight, upper_weight = \
        map(int, item[["weight", "lower_weight", "upper_weight"]].tolist())
    value, item_id = item["value"], int(item.name)

    if len(paths) == 0:
        state[0, weight] = value
        paths[weight] = {item_id}
        return

    temp_state = lil_matrix(state, copy=True)
    temp_paths = paths.copy()

    # 2 domains
    values, weights = state.data[0], state.rows[0]

    # Initial index (= weight)
    lowest_index = min([w for w in weights
                       if w <= lower_weight - weight] or [0])

    cmpr_q = deque()
    # Initial values
    cmpr_v = state[0, lowest_index] + value
    cmpr_w = max(lower_weight, 0 + weight)
    cmpr_path = paths.get(lowest_index, set()) | {item_id}

    cur_max = 0
    for w, v in zip(weights, values):
        # Accumulate possible substitutes
        if lower_weight <= w + weight <= upper_weight:
            cmpr_q.append((w + weight, v + value))


        if lower_weight <= w <= upper_weight:
            # Go through all items in cmpr_q less than w
            while cmpr_w < w:
                # If cmpr_v > cur_max insert item, update cur_max
                # skip it in other case
                if cmpr_v > cur_max:
                    # --> New item
                    temp_state[0, cmpr_w] = cur_max = cmpr_v
                    temp_paths[cmpr_w] = cmpr_path
                # Take next item from queue in any case
                try:
                    cmpr_w, cmpr_v = cmpr_q.popleft()
                    cmpr_path = paths.get(cmpr_w - weight) | {item_id}
                except IndexError as e:
                    break
            # If cmpr_w == w and cmpr_v > v, insert cmpr_v
            # and update cur_max
            if cmpr_w == w and cur_max < cmpr_v > v:
                # --> New item
                temp_state[0, cmpr_w] = cur_max = cmpr_v
                temp_paths[cmpr_w] = cmpr_path
            elif v > cur_max:
                cur_max = v
    # Fill rest
    while True:
        if cur_max < cmpr_v:
            # --> New item
            temp_state[0, cmpr_w] = cur_max = cmpr_v
            temp_paths[cmpr_w] = cmpr_path
        try:
            cmpr_w, cmpr_v = cmpr_q.popleft()
            cmpr_path = paths.get(cmpr_w - weight) | {item_id}
        except IndexError as e:
            break

    # Apply changes
    for value, weight in zip(temp_state.data[0], temp_state.rows[0]):
        if lower_weight <= weight <= upper_weight:
            state[0, weight] = value
            paths[weight] = temp_paths[weight]
        elif weight in paths:
            # Replace values not in window
            state[0, weight] = 0
            paths.pop(weight)

def get_result(items):
    """ Return sum of values of taken items """
    return items.loc[items["take"] == 1, "value"].astype(int).sum()


class Knapsack(object):
    def __init__(self, capacity=0, items=None):
        self.result = 0

        self.capacity = capacity
        self.items = prepare_items(items)

        n_items = items.shape[0] if isinstance(items, pd.DataFrame) else 0
        self.n_items = n_items
        prune_zero_values(self)
        self.reset_filled_space()
        prune_exceeded_free_space(self)

        # Aux
        self.paths = dict()
        self.state = lil_matrix((1, self.capacity + 1))

    def __repr__(self):
        return "Knapsack. Capacity: {}/{}, items: {}".format(
            self.filled_space, self.capacity, self.n_items)

    def feasibility_check(self):
        """ Check if total weight of taken items is
            less than knapsack capacity """
        assert self.items.loc[self.items["take"] == 1, "weight"].sum() \
            <= self.capacity, "Not feasible answer. Exceeded capacity."

    def reset_filled_space(self):
        """ Return free space after taking items """
        used_space = self.items.loc[self.items["take"] == 1, "weight"].sum()
        self.filled_space = int(used_space)

    def get_result(self):
        """ Return sum of values of taken items """
        return get_result(self.items)

    def get_answer(self):
        """ Return answer as sequence of zeros and ones """
        return self.items.sort_index()["take"].astype(int).tolist()

    def eval_left(self, param="value", item_id=None):
        """ Return sum of param for untouched items """
        item_id = item_id or 0
        columns = [param, "take"]
        tmp = self.items.loc[item_id:]
        return tmp.loc[tmp["take"].isnull(), param].sum()

    def take_item(self, item_id):
        """ Set take == 1 & reduce paths """
        self.items.loc[item_id, "take"] = 1
        self.reset_filled_space()
        self.calculate_boundaries(item_id)
        item = self.items.loc[item_id, :]
        logging.info("Item in take_item:\n{}".format(item))

        weight = int(item["weight"])
        for w in tuple(self.paths):
            if item["lower_weight"] <= w <= item["upper_weight"]:
                continue
            else:
                logging.debug("Remove w: {}".format(w))
                self.state[0, w] = 0
                self.paths.pop(w)

    def reset_dp(self):
        aux_columns = [
            "avail_weight",
            "upper_weight",
            "lower_weight",
            "avail_value",
            "max_val",
            "min_ix",
        ]
        for w in aux_columns:
            self.items[w] = None

    def calculate_taken(self, value="value"):
        """ Calculate total of taken items values (value or weight) """
        return self.items.loc[self.items["take"] == 1, value].sum()

    def calculate_boundaries(self, item_id):
        """ Calculate useful values for taken item """
        item = self.items.loc[item_id, :].copy()
        # all_taken = self.items.loc[:item_id, "weight"].sum() FAIL
        for param in "weight", "value":
            item["avail_%s" % param] = self.eval_left(param, item_id)

        item["max_val"] = self.state.tocsr().max()
        low_val = max(0, item["max_val"] - item["avail_value"])
        if low_val:
            item["min_ix"] = min((self.state >= low_val).tolil().rows[0])
        else:
            item["min_ix"] = 0

        item["upper_weight"] = self.capacity
        item["lower_weight"] = max(self.filled_space, item["min_ix"],
            self.capacity - self.eval_left("weight", item_id))

        self.items.loc[item_id, :] = item

    def prune(self):
        """ Prune as method """
        return prune(self)

    def forward(self):
        """ Fill domain """
        self.reset_dp()
        search_items = self.items.loc[self.items["take"].isnull()]

        prune_freq = min(self.n_items // 10 + 1, 100)
        for order, (cur_id, item) in enumerate(search_items.iterrows()):
            if ~self.items.isnull().loc[cur_id, "take"]:
                continue
            logging.info("Forward. order: {}".format(order, cur_id))

            self.calculate_boundaries(cur_id)
            item = self.items.loc[cur_id]
            forward_step(item, self.state, self.paths)
            if order % prune_freq == 0 and \
                    0 < item["lower_weight"] < self.capacity - \
                    self.items.loc[cur_id, "weight"]:
                self.prune()
            self.feasibility_check()

    def backward(self):
        """ Find answer """
        # Max value
        logging.info("Backward stage")
        ix = int(self.state.tocsr()[0,:].argmax())
        logging.info("Result ix: {}".format(ix))
        logging.info("Path:\n{}".format(self.paths[ix]))
        for item_id in self.paths[ix]:
            self.items.loc[item_id, "take"] = 1
        self.items["take"].fillna(0, inplace=True)
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

if __name__=="__main__":
    pass
