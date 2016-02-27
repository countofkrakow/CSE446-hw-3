import csv

def load_csv(filename):
    lines = []
    with open(filename) as csvfile:
        reader = csv.DictReader(csvfile)
        for line in reader:
            lines.append(line)
    return lines

def load_test_data():
    return load_csv("..\\test.csv")

#TODO: Possibly use different data for training and validation
def load_train_data():
    return load_csv("..\\validation.csv")