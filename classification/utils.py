import csv
import pandas as pd
import numpy as np


def write_submission(id, genre):

    for i in range(len(id)):
        id[i] = int(id[i][:-4])

    df = pd.DataFrame({"id": id, "genre": genre})
    df.to_csv("submit.csv", index=False)


def get_classes(filepath, pad=True):
    with open(filepath) as f:
        reader = csv.reader(f)
        data = list(reader)
    # TODO figure out way to get rid of header row
    if pad:
        for i in range(len(data)):
            temp = f"{int(data[i][0]):08d}"
            data[i][0] = "{0:s}.wav".format(temp)
    else:
        for i in range(len(data)):
            data[i][0] = "{0:s}.wav".format(data[i][0])

    print('padding', pad)
    print(data[0])

    return data


def get_ids(filepath):
    """gets ids with now padding for test submission"""
    with open(filepath) as f:
        reader = csv.reader(f)
        data = list(reader)

    ids = np.array(data)[:, 0]
    return ids
