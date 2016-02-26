class perceptron:

    def __init__(self, kern=None):
        self.kern = kern
        if kern is None:
            self.kern = self.dot_prod


    def dot_prod(self, x, y):


    def train(self, data):
        pass

    def classify(self, point):
        pass

def extract_features(raw):
    data = []
    for r in raw:
        point = {}
        point["label"] = (r['income'] == '>50K')

        features = []
        features.append(1.)
        features.append(float(r['age'])/100)
        features.append(float(r['education_num'])/20)
        features.append(float(r['capital_gain']) / 1000)
        features.append(float(r['hr_per_week'])/20)
        features.append(r['country'] == 'United-States')
        features.append(r['type_employer'] == 'Private')
        features.append(r['type_employer'] == 'Federal-gov')
        features.append(r['education'] == 'HS-grad')
        features.append(r['marital'] == 'Never-married')
        features.append(r['relationship'] == 'Husband')
        features.append(r['relationship'] == 'Wife')
        features.append(r['sex'] == 'Female')
        point['features'] = features
        data.append(point)
    return data