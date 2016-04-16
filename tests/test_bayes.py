"""NaiveBayes tests."""

import unittest

from traintestsets import TrainTestSets
from bayes import NaiveBayes


class TestNaiveBayes(unittest.TestCase):
    """Unittest of NaiveBayes."""

    mushrooms_path = 'test_data/mushrooms.mff'
    votes_path = 'test_data/votes.mff'

    def setUp(self):
        """Setup unittest."""
        self.traintestsets = TrainTestSets()
        pass

    def test_(self):
        """Run basic test."""
        self.assertEqual('foo'.upper(), 'FOO')

    def test_isupper(self):  # noqa
        self.assertTrue('FOO'.isupper())
        self.assertFalse('Foo'.isupper())

    def test_split(self):  # noqa
        s = 'hello world'
        self.assertEqual(s.split(), ['hello', 'world'])
        # check that s.split fails when the separator is not a string
        with self.assertRaises(TypeError):
            s.split(2)

if __name__ == '__main__':
    unittest.main()  # noqa