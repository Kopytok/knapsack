import pandas as pd

from sparse_dynamic_programming import Knapsack

def main():
    """ Usage example """
    items = pd.read_csv("200_items.csv")
    capacity = 2640230

    knapsack = Knapsack(capacity, items)
    answer = knapsack.solve()
    print("Total value of selected items: {}".format(knapsack.result))

if __name__=="__main__":
    main()
