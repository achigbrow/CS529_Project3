import csv


def get_classes(filepath):
    with open(filepath) as f:
        reader = csv.reader(f)
        data = list(reader)

    for i in range(len(data)):
        temp = f"{int(data[i][0]):08d}"
        data[i][0] = "{0:s}.wav".format(temp)

    print(data[0])

    return data


if __name__ == "__main__":
    data = get_classes(r"D:\proj3_data\project3\train.csv")
    # data = get_classes(r"D:\repos\CS529_Project3\train1.csv")
