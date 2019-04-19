from .imports import *

def prepare_items(items=None, by=["value", "density",], ascending=False):
    """ Convert list of namedtuples into dataframe and sort it """
    if isinstance(items, pd.DataFrame):
        items["density"] = items.eval("value / weight")
        if not by:
            dens_first = (items["density"].std() > 0)
            by = "density" if dens_first else "value"
            else_by = "density" if by != "density" else "value"
            by = [by, else_by]
            # ascending = [True, False] if dens_first else [False, True]
        items.sort_values(by, ascending=ascending, inplace=True)
        logging.info("Sorted by {}".format(by))
        items["take"] = None
        logging.info("First 5 items:\n{}".format(items.head()))
        return items
    return pd.DataFrame(columns=["value", "weight", "density", "take"])

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
