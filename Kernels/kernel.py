import operator

class perceptron:
    def __init__(self, data, kern=None, trainingSteps=100):
        if kern is None:
            self.kern = self.dot_prod
        else:
            self.kern = kern
        self.steps = trainingSteps
        self.data = extract_features(data)

    def dot_prod(self, x, y):
        assert len(x) == len(y)
        return sum(map(operator.mul, x, y)

    def train(self):
        lastMistake = 0 # used for exiting early when we converge
        self.a = [0]*len(self.data)
        for i in range(self.steps): # num iterations
            imod = i % len(self.data)
            x_i = self.data[imod]['features']
            y_i = self.data[imod]['label']

            if not self.predict(x_i):
                # we've hit an error
                lastMistake = imod
                if sign(y_i):
                    self.a[imod] += 1
                else:
                    self.a[imod] -= 1

            elif imod == lastMistake and i != 0: # converged
                print("Training converged at iteration %d" % i)
                break

    def predict(self, point):
        pred = 0
        for j in range(len(self.data)):
            pred += self.a[j] * self.kern(point['features'], self.data[j]['features'])
        return sign(pred) == sign(point['label'])

# True for x > 0
# False for x <= 0
def sign(x):
    return x > 0

def extract_features(raw):
    data = []
    for r in raw:
        point = {}
        point["label"] = (r['valy'])
        features = []
        for i in range(len(r)-1):
            features.append(r['pixel' + str(i)])
        point['features'] = features
        data.append(point)
    return data