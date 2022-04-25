import csv

def write_submission(id, genre):


def get_classes(filepath):
  with open(filepath) as f:
    reader = csv.reader(f)
    data = list(reader)

  for i in range(len(data)):
    temp = f"{int(data[i][0]):08d}"
    data[i][0] = "{0:s}.wav".format(temp)

  print(data[0])

  return data