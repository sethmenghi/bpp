"""
example.py.

Holds the Example and Examples classes.
"""
import all_exceptions as exceptions
from attribute import NominalAttribute, NumericAttribute
from bayes import mean, standard_deviation


class Example(object):
    """Stores the attribute values of an example.

    Numeric values are stored as is. Nominal values are stored as doubles
    and are indices of the value in the attributes structure.

    Attributes
    ----------
        _n (int): number of examples
        _values (list): list of values in this example
    """

    _n = 0

    def __init__(self, n=0):
        """Explicit constructor, `n` is the number of examples."""
        self._n = n
        self._values = []

    def __iter__(self):
        """Make iterable."""
        return iter(self.values)

    def __getitem__(self, val):
        """Allow for indexing."""
        return self.values[val]

    def __str__(self):
        """String representation fo example."""
        return str(self.values)

    @property
    def values(self):  # noqa
        return self._values

    @property
    def size(self):
        """Return length of self.values."""
        return len(self.values)

    def append(self, val):
        """Append a value onto self._values."""
        self._values.append(val)
        self._n = len(self._values)


class Examples(object):
    """Stores examples for data sets for machine learning.

    Attributes
    ----------
        _attributes (Attributes):  attributes.Attributes object
        _examples (list): list of Example objects
    """

    _attributes = None

    def __init__(self, attributes=None):
        """Constructor, `attributes` (default=None), Attributes() object."""
        self._attributes = attributes
        self._examples = []

    def __iter__(self):
        """Make iterable."""
        return iter(self._examples)

    def __getitem__(self, val):
        """Allow for indexing."""
        return self._examples[val]

    def __len__(self):
        """Allow for len() on an Examples object."""
        return self.size

    def get_class_value_at(self, i):
        """Return class value of example at i."""
        return self._examples[i][self.attributes.classindex]

    @property
    def attributes(self):
        """Return attributes."""
        return self._attributes

    @attributes.setter
    def attributes(self, val):
        """Set attributes."""
        self._attributes = val

    @property
    def examples(self):
        """Return list of examples."""
        return self._examples

    @examples.setter
    def examples(self, val):
        """Set list of examples."""
        self._examples = val

    @property
    def size(self):
        """Return length of size."""
        return len(self._examples)

    def append(self, example):
        """Append an example onto the example list."""
        self._examples.append(example)

    def parse(self, line):
        """Given the attributes structure, parses into Examples."""
        values = line.split(" ")
        example = Example()
        for i, v in enumerate(values):
            if i >= self._attributes.size:
                e = "Out of bounds for val %s in Examples.parse." % v
                raise exceptions.LogicError(e)
            if isinstance(self._attributes[i], NumericAttribute):
                val = float(v)
            elif isinstance(self._attributes[i], NominalAttribute):
                val = self._attributes[i].index(v)
            else:
                e = "No attribute type for %s in Examples.parse()" % self._attributes[i]
                raise exceptions.LogicError(e)
            example.append(val)
        self.append(example)

    def mean(self, attributeindex):
        """Return mean of all example's attribute at attributeindex."""
        return mean([e.values[attributeindex] for e in self.examples])

    def stdev(self, attributeindex):
        """Return stdev of example's attribute at attributeindex."""
        return standard_deviation([float(e.values[attributeindex]) for e in self.examples])

    def __str__(self):
        """Return string reprsentation of Examples."""
        # self._examples.values
        string = ""
        for e in self._examples:
            for i, v in enumerate(e.values):
                if isinstance(self._attributes[i], NominalAttribute):
                    v = int(v)
                    string = string + str(self._attributes[i].domain[v])
                else:
                    string = string + str(v)
                if i == len(e.values) - 1:
                    string = string + "\n"
                else:
                    string = string + " "
        return string
