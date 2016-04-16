#!/usr/bin/env python # noqa
# -*- coding: utf-8 -*-
#
# Name: Seth menghi
# E-mail Address: swm36@georgetown.edu
# Platform: MacOS
# Language/Environment: python
#
# In accordance with the class policies and Georgetown's Honor Code,
# I certify that, with the exceptions of the class resources and those
# items noted below, I have neither given nor received any assistance
# on this project.
#
import argparse

import all_exceptions as exceptions
from neuralnetwork import NeuralNetwork
from traintestsets import TrainTestSets
from evaluator import Evaluator


def parse_args():
    """Parse arguments with argparse."""
    parser = argparse.ArgumentParser(description="Load a test set into TrainTestSet.")
    parser.add_argument('-t', '--trainfile', dest="trainfile",
                        help="trainfile that contains the data, .mff format")
    parser.add_argument('-T', '--testfile', dest="testfile", default=None,
                        help="testfile that contains the data .mff format")
    parser.add_argument('-p', '--holdout', dest="holdout", default=None, type=float,
                        help="holdout proportation for the user")
    parser.add_argument('-x', '--folds', dest="folds", default=10, type=int,
                        help="number of folds for cross validation")
    parser.add_argument('-e', '--min_error', dest="min_error", default=.80, type=float,
                        help='minimum error threshold for the backprop')
    parser.add_argument('-n', '--learningrate', dest='n', default=.01, type=float,
                        help='learning rate for the backprop algorithm')
    parser.add_argument('-j', '--hiddennodes', dest='j', type=int,
                        help='number of hidden nodes for the backprop algorithm')
    parser.add_argument('-d', '--debug', dest='debug', default=False,
                        help='print debug statements: defaults=False')
    parser.add_argument('-z', '--test', dest="test", default=False,
                        help='For testing purposes IGNORE')
    args = parser.parse_args()
    return args


def create_dataset(trainfile, testfile=None):
    """Input file of test set."""
    if testfile is None and trainfile is None:
        raise exceptions.LogicError("No trainfile supplied")
    traintestsets = TrainTestSets(test_path=testfile, train_path=trainfile)
    traintestsets.train.normalize_attributes()
    return traintestsets


def evaluate(classifier, testset=None, holdout=None, folds=10):
    """Create evaulator object.

    Args:
        classifier (Classifier): desired classifier to run
        testset (DataSet): testset to run classification accuracies/tests
        outfile (str): filepath of target output file
        holdout (float): desired split for the hold-out method
        folds (int): number of folds for cross validation
    """
    evaluator = Evaluator(classifier)
    if testset:
        pass
    elif holdout:
        evaluator.holdout(holdout)
    else:  # runing folds
        evaluator.cross_validate(folds)
    return evaluator


def _test():
    dset = create_dataset('tests/lenses.mff')
    dset.train.normalize_attributes()
    for e in dset.train.examples:
        print(e)
    classifier = NeuralNetwork(trainset=dset.train, max_error=.2, debug=True)
    evaluator = Evaluator(classifier)
    evaluator.holdout(.5)

    dset = create_dataset('tests/lenses.mff')
    dset.train.nominal_to_linear()
    print(dset)
    classifier = NeuralNetwork(trainset=dset.train, debug=True, max_error=.1, j=10)
    evaluator = Evaluator(classifier)
    evaluator.holdout(.2)
    dset = create_dataset('tests/test_data/iris-binary.mff')
    classifier = NeuralNetwork(trainset=dset.train, debug=True, max_error=.1)
    classifier.train(dset.train)
    dset = create_dataset('tests/test_data/votes.mff')
    classifier = NeuralNetwork(trainset=dset.train, debug=True, max_error=.1)
    classifier.train(dset.train)
    dset = create_dataset('tests/test_data/mushroom.mff')
    classifier = NeuralNetwork(trainset=dset.train, debug=True)
    classifier.train(dset.train)
    dset = create_dataset('tests/test_data/soybean.mff')
    classifier = NeuralNetwork(trainset=dset.train, debug=True)
    classifier.train()


def main():
    """Main function, runs all the above functions."""
    args = parse_args()
    if args.test:
        _test()
    else:
        dataset = create_dataset(args.trainfile, args.testfile)
        max_error = 1.0 - args.min_error
        # self, trainset, n=0.01, j=5, max_error=.3, debug=False
        classifier = NeuralNetwork(dataset.train, args.n, args.j, max_error, debug=args.debug)
        evaluate(classifier, dataset.test_set, args.holdout, args.folds)


if __name__ == '__main__':
    main()
