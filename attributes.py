"""
attributes.py.

Holds the Attributes class.
"""
import logging

import all_exceptions as exceptions
from attribute import NominalAttribute, NumericAttribute, Attribute


logger = logging.getLogger(__name__)


class Attributes(object):
    """Stores information for attributes for data sets."""

    def __init__(self):
        """Intialization of Attributes object."""
        self.attributes = []
        self._has_nominal_attribute = False
        self._has_numeric_attribute = False
        # self._classindex = -1
        self._classindex = -1

    def __iter__(self):
        """Make iterable."""
        return iter(self.attributes)

    def __getitem__(self, val):
        """Allow for indexing."""
        if isinstance(val, str):
            return self.attributes.index(val)
        return self.attributes[val]

    def __str__(self):
        """String representation of Attributes."""
        string = ""
        for a in self.attributes:
            string = string + str(a) + "\n"
        return string

    @property
    def size(self):
        """Return number of attributes."""
        return len(self.attributes)

    @property
    def classindex(self):
        """Return the class index attribute."""
        return self.size - 1

    @property
    def class_size(self):
        """Return the class size (the domain of the class)."""
        if isinstance(self.attributes[self.classindex], Attribute):
            return 1
        else:  # multiple attributes make up class
            return len(self.attributes[self.classindex])

    @property
    def has_numeric_attribute(self):
        """Return True if self.attributes has numeric attribute."""
        return self._has_numeric_attribute

    @property
    def has_nominal_attribute(self):
        """Return true if self.attributes has nominal attribute."""
        return self._has_nominal_attribute

    def add(self, attribute):
        """Add a new attribute to this set of attributes."""
        if isinstance(attribute, NominalAttribute) and not self.has_nominal_attribute:
            self._has_nominal_attribute = True
        elif isinstance(attribute, NumericAttribute) and not self.has_numeric_attribute:
            self._has_numeric_attribute = True
        self.attributes.append(attribute)

    def parse(self, line):
        """Parse a string attribute line."""
        if '@attribute' in line:
            attributes = line.split(" ")
            if attributes[2] == 'numeric':
                attribute = self._parse_numeric_attribute(attributes)
            else:
                attribute = self._parse_nominal_attribute(attributes)
            self.add(attribute=attribute)
            self._classindex += 1
        else:
            e = "Line is not an attribute."
            raise exceptions.LogicError(e)

    def _parse_numeric_attribute(self, attributes):
        if len(attributes) != 3:
            e = "Numeric attribute w/ incorrect num of args (%s != 3)." % len(attributes)
            raise exceptions.LogicError(e)
        attribute = NumericAttribute(name=attributes[1])
        return attribute

    def _parse_nominal_attribute(self, attributes):
        if len(attributes) < 2:
            e = "Nominal attribute with no domain."
            raise exceptions.LogicError(e)
        attribute = NominalAttribute(name=attributes[1])
        for value in attributes[2:]:
            attribute.add_value(value=value)
        return attribute


class AttributeFactory(object):
    """Processes a single attribute declaration."""

    attributes = Attributes()

    def make(self, line):
        """Process line into attributes."""
        self.attributes.parse(line)

    def __iter__(self):
        """Make iterable."""
        return iter(self.attributes)

    def __getitem__(self, val):
        """Allow for indexing."""
        return self.attributes[val]
