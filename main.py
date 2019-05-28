import pandas as pd

from knapsack.imports import *
from knapsack.open_data import select_file_in
from knapsack.knapsack import Knapsack

logging.basicConfig(level=logging.INFO,
    format="%(levelname)s - %(asctime)s - %(msg)s",
    datefmt="%Y-%m-%d %H:%M:%S",
    handlers=[
#        logging.FileHandler("log_knapsack.log"), # For debug
        logging.StreamHandler(),
    ])

def main():
    """ Usage example """
    path = select_file_in("data")
    knapsack = Knapsack.load(f"data/{path}")
    answer = knapsack.solve()
    print("Total value of selected items: {}".format(knapsack.result))

if __name__=="__main__":
    main()
