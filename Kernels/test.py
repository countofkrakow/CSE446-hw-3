import unittest
import operator
from math import exp, sqrt
from data import load_test_data, load_train_data
from kernel import perceptron, extract_features

class kernelTest(unittest.TestCase):

    def setUp(self):
        self.train_data = extract_features(load_train_data())
        self.test_data = extract_features(load_test_data())

    def exp_kern(self, x, y, deg=1):
        assert len(x) == len(y)
        abs_sum = 0
        for i in range(len(x)):
            abs_sum += (x[i] - y[i])**2
        return exp(-sqrt(abs_sum) / (2*(deg**2)))

    def dot_prod_poly(self, x, y):
        return self.dot_prod(x, y) + 1

    def dot_prod(self, x, y):
        assert len(x) == len(y)
        return sum(map(operator.mul, x, y))

    def run_test(self, p, train_data, test_data, lossInterval=None, iterations=1000):
        p.train(lossInterval=lossInterval, iterations=iterations)
        correct = 0
        for point in train_data:
            if p.predict(point):
                correct += 1
        print('Training data loss: %f' % (1 - float(correct) / len(train_data)))

        correct = 0
        for point in test_data:
            if p.predict(point):
                correct += 1
        print('Test data loss: %f' % (1 - float(correct) / len(test_data)))
        print('\n')

    # Q3.1
    #
    # n = 1 case where we plot loss every 100 steps
    def test_dotProdKernel(self):
        print('Testing n = 1 degree polynomial kernel after 1 pass through training data:')
        print('-'*75)
        p = perceptron(self.train_data, lambda x, y: self.dot_prod_poly(x, y))
        self.run_test(p, self.train_data, self.test_data, lossInterval=100, iterations=len(self.train_data))

    # Q3.2
    # tries out all of the different polynomial kernels
    #
    #@unittest.skip("demonstrating skipping")
    def test_polyKernels(self):
        print('Testing polynomial kernels of varying degrees:')
        print('-'*75)
        print('Test n = 1 degree polynomial kernel after 1000 steps:')
        p = perceptron(self.train_data, lambda x, y: self.dot_prod_poly(x, y))
        self.run_test(p, self.train_data, self.test_data)

        print('Test n = 3 degree polynomial kernel after 1000 steps:')
        p = perceptron(self.train_data, lambda x, y: self.dot_prod_poly(x, y)**3)
        self.run_test(p, self.train_data, self.test_data)

        print('Test n = 5 degree polynomial kernel after 1000 steps:')
        p = perceptron(self.train_data, lambda x, y: self.dot_prod_poly(x, y)**5)
        self.run_test(p, self.train_data, self.test_data)

        print('Test n = 7 degree polynomial kernel after 1000 steps:')
        p = perceptron(self.train_data, lambda x, y: self.dot_prod_poly(x, y)**7)
        self.run_test(p, self.train_data, self.test_data)

        print('Test n = 10 degree polynomial kernel after 1000 steps:')
        p = perceptron(self.train_data, lambda x, y: self.dot_prod_poly(x, y)**10)
        self.run_test(p, self.train_data, self.test_data)

        print('Test n = 15 degree polynomial kernel after 1000 steps:')
        p = perceptron(self.train_data, lambda x, y: self.dot_prod_poly(x, y)**15)
        self.run_test(p, self.train_data, self.test_data)

        print('Test n = 20 degree polynomial kernel after 1000 steps:')
        p = perceptron(self.train_data, lambda x, y: self.dot_prod_poly(x, y)**20)
        self.run_test(p, self.train_data, self.test_data)

    # Q3.3
    # Empirical testing on the different degree polynomial kernels
    # shows that 3 is the optimal degree for poly kernels
    #
    #@unittest.skip("")
    def test_exponentialKernel(self):
        print('Testing exponential kernel against polynomial kernel:')
        print('-'*75)
        print('Test n = 3 degree polynomial kernel after 1000 steps:')
        p = perceptron(self.train_data, lambda x, y: self.dot_prod_poly(x, y,)**3)
        self.run_test(p, self.train_data, self.test_data, lossInterval=100)

        print('Test sigma = 10 exponential kernel after 1000 steps:')
        p = perceptron(self.train_data, lambda x, y: self.exp_kern(x, y, 10))
        self.run_test(p, self.train_data, self.test_data, lossInterval=100)

if __name__ == '__main__':
    unittest.main()