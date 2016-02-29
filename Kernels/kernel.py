
class perceptron:
    def __init__(self, data, kern):
        self.kern = kern
        self.data = data

    def train(self, iterations=1000, lossInterval=None):
        wrong = 0
        self.mistakes = []
        for i in range(iterations): # num iterations

            imod = i % len(self.data)
            y_i = self.data[imod]['label']
            if not self.predict(self.data[imod]):
                wrong += 1
                self.mistakes.append((self.data[imod]['features'], y_i))

            if lossInterval is not None and i % lossInterval == 0 and i != 0:
                print('average loss at %d steps: %f' % (i, float(wrong)/(i)))

    def predict(self, point):
        pred = 0
        for x, y in self.mistakes:
            pred += y * self.kern(point['features'], x)
        return sign(pred) == sign(point['label'])

# True for x > 0
# False for x <= 0
def sign(x):
    return x > 0

def extract_features(raw):
    data = []
    for r in raw:
        point = {}
        point["label"] = int(r['label'])
        features = []
        for j in range(len(r)-1):
            features.append(int(r['pixel' + str(j)]))
        point['features'] = features
        data.append(point)
    return data