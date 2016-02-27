import unittest
from data import load_test_data, load_train_data
from kernel import perceptron, extract_features

class kernelTest(unittest.TestCase):
    def test_dotProdKernel(self):
        train_data = extract_features(load_train_data())
        test_data = extract_features(load_test_data())
        p = perceptron(train_data)
        p.train(50)
        print p.a

        correct = 0
        for point in train_data:
            if p.predict(point):
                correct += 1
        print('Training data accuracy %f' % (float(correct) / len(train_data)))

        correct = 0
        for point in test_data:
            if p.predict(point):
                correct += 1
        print('Test data accuracy %f' % (float(correct) / len(test_data)))

if __name__ == '__main__':
    unittest.main()