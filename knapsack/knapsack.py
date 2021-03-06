from .imports import *
from .prune import *
from .open_data import *
from .dynamic_programming import *


class Knapsack(object):
    def __init__(self, capacity=0, items=None):
        self.result = 0

        self.capacity = capacity
        self.items = prepare_items(items)

        self.dp = DPForward(capacity)

        n_items = items.shape[0] if isinstance(items, pd.DataFrame) else 0
        self.n_items = n_items
        prune_zero_values(self)
        self.reset_filled_space()
        prune_exceeded_free_space(self)

    def __repr__(self):
        return f"Knapsack. Capacity: {self.filled_space}/{self.capacity}, " +\
            f"items: {self.n_items}"

    def feasibility_check(self):
        """ Check if total weight of taken items is
            less than knapsack capacity """
        assert self.items.loc[self.items["take"] == 1, "weight"].sum() \
            <= self.capacity, "Not feasible answer. Exceeded capacity."

    def reset_filled_space(self):
        """ Return free space after taking items """
        used_space = self.items.loc[self.items["take"] == 1, "weight"].sum()
        self.filled_space = int(used_space)

    def get_result(self, items=None):
        """ Return sum of values of taken items """
        items = items or self.items
        return items.loc[items["take"] == 1, "value"].astype(int).sum()

    def get_answer(self):
        """ Return answer as sequence of zeros and ones """
        return self.items.sort_index()["take"].astype(int).tolist()

    def eval_left(self, param="value", item_id=None):
        """ Return sum of `param` for untouched items """
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
        logging.info(f"Item in take_item:\n{item}")

        for w in tuple(self.dp.paths):
            if item["lower_weight"] <= w <= self.capacity:
                continue
            else:
                logging.debug(f"Remove w: {w}")
                self.dp.state[0, w] = 0
                self.dp.paths.pop(w)

    def reset_dp(self):
        aux_columns = [
            "avail_weight",
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
        for param in "weight", "value":
            item["avail_%s" % param] = self.eval_left(param, item_id)

        item["max_val"] = self.dp.state.tocsr().max()

        low_val = max(0, item["max_val"] - item["avail_value"])
        item["min_ix"] = min((self.dp.state >= low_val).tolil().rows[0]) \
            if low_val else 0

        item["lower_weight"] = max(
            self.filled_space,
            item["min_ix"],
            self.capacity - self.eval_left("weight", item_id)
        )

        self.items.loc[item_id, :] = item

    def prune(self):
        """ Prune as method """
        return prune(self)

    def forward(self):
        """ Fill domain """
        self.reset_dp()
        search_items = self.items.loc[self.items["take"].isnull()]

        self.dp = DPForward(self.capacity)
        prune_freq = min(self.n_items // 10 + 1, 100)
        for step, (cur_id, item) in enumerate(search_items.iterrows()):
            if ~self.items.isnull().loc[cur_id, "take"]:
                continue
            self.calculate_boundaries(cur_id)
            item = self.items.loc[cur_id]
            item_data = ' - '.join([f"{k}: {v}"
                for k, v in item.to_dict().items()])
            logging.info(f"\nStep: {step} - {item_data}")

            self.dp.forward_step(item)
            if step % prune_freq == 0 and \
                    0 < item["lower_weight"] < self.capacity - \
                    self.items.loc[cur_id, "weight"]:
                self.prune()
            self.feasibility_check()

    def backward(self):
        """ Find answer """
        ix = int(self.dp.state.tocsr()[0,:].argmax())
        logging.info(f"Result ix: {ix}")
        logging.info(f"Path:\n{self.dp.paths[ix]}")

        for item_id in self.dp.paths[ix]:
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
        logging.info("Finished backward. Total time (sec): "
                     f"{time.time() - t0}")

        self.feasibility_check()
        answer = self.get_answer()
        logging.info(f"Resulting value: {self.result}")
        logging.info(f"Selected items: {answer}")
        return answer

    @classmethod
    def load(cls, path):
        """ Load knapsack from file """
        capacity, items = read_knapsack(path)
        knapsack = cls(capacity, items)
        return knapsack


if __name__ == "__main__":
    pass
