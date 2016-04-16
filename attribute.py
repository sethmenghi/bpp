"""
attributes.py.

Holds the Attribute, NominalAttribute, and NumericAttribute classes.
"""
import logging

import all_exceptions as exceptions


logger = logging.getLogger(__name__)


class Attribute(object):
    """Stores information for an attribute."""

    domain = []

    def __init__(self, name=None):
        """Constructor for the Attribute object.

        Args:
        name (str): name of the attribute.
            default=None
        """
        self._name = name
        self.domain = []

    @property
    def name(self):
        """Return name attribute."""
        return self._name

    @name.setter
    def name(self, name):
        """Set the name attribute to `name`."""
        self._name = name

    def __repr__(self):
        """Representation of the object."""
        return self.name

    def split_into_binary(self):
        """Convert current attribute to binary encoding."""
        if len(self.domain) < 2:
            yield self
        else:
            for i, v in enumerate(self.domain):
                attribute = BinaryAttribute(name=self.name)
                attribute.domain = [0, 1]
                attribute.place = i
                attribute.value = v
                yield attribute


class BinaryAttribute(Attribute):
    """Represents an attribute where the with a domain of more than 1."""

    @property
    def place(self):
        """Place of the binary digit."""
        return self._place

    @place.setter
    def place(self, val):
        self._place = val

    @property
    def value(self):
        """Actual value that this encoding represents."""
        return self._value

    @value.setter
    def value(self, val):
        self._value = val


class NominalAttribute(Attribute):
    """An attribute with nominal representation."""

    domain = []

    @property
    def size(self):
        """Return size attribute."""
        return len(self.domain)

    def __iter__(self):
        """Make iterable."""
        return iter(self.domain)

    def __getitem__(self, val):
        """Allow for indexing."""
        return self.domain[val]

    def __str__(self):
        """The string repr. of an Attribute is it's name."""
        string = "@attribute %s" % self.name
        for v in self.domain:
            string = string + " %s" % v
        return string

    def add_value(self, value):
        """Add a new nominal value to the domain of this nominal attribute."""
        try:
            if value not in self.domain:
                self.domain.append(value)
        except:
            e = 'Error with adding value: %s.' % value
            raise exceptions.LogicError(e)

    def index(self, value):
        """Return the index of the `value` for this nominal attribute."""
        for i, k in enumerate(self):
            if k == value:
                return i


class NumericAttribute(Attribute):
    """Store information for a numeric attribute.

    A numeric attribute has a name. Its domain is the real numbers.
    """

    domain = ['numeric']

    def __str__(self):
        """Numeric attribute string representation."""
        return "@attribute %s numeric" % self.name
