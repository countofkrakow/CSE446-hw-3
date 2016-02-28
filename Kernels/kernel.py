
class perceptron:
    def __init__(self, data, kern):
        self.kern = kern
        self.data = data

    def train(self, iterations=1000, lossInterval=None):
        wrong = 0
        self.a = [0]*len(self.data)
        for i in range(iterations): # num iterations

            imod = i % len(self.data)
            y_i = self.data[imod]['label']
            if not self.predict(self.data[imod]):
                wrong += 1
                if sign(y_i):
                    self.a[imod] += 1
                else:
                    self.a[imod] -= 1

            if lossInterval is not None and i % lossInterval == 0:
                print 'average loss at %d steps: %f' % (i, float(wrong)/(i+1))

    def predict(self, point):
        pred = 0
        for j in range(len(self.data)):
            pred += self.a[j] * self.kern(point['features'], self.data[j]['features'])
        ret = sign(pred) == sign(point['label'])
        return ret

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