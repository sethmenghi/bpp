"""
dataset.py.

Holds the DataSet class.
"""
from example import Examples
from math import log
from operator import itemgetter
from collections import defaultdict
from attribute import NominalAttribute, NumericAttribute
from attributes import Attributes


class DataSet(object):
    """Implements a class for a data set for machine-learning methods."""

    def __init__(self, attributes=None, examples=None, name=None):
        """Constructor, default attributes argument=None."""
        self._attributes = attributes
        self._examples = Examples(self._attributes)
        self._name = name
        self.add(examples=examples)

    def __str__(self):
        """Return string representation of dataset as would look in file."""
        string = "@dataset %s\n\n" % self._name
        string = string + str(self.attributes) + "\n"
        string = string + "@examples\n\n"
        string = string + str(self.examples)
        return string

    @property
    def attributes(self):
        """Getter for _attributes."""
        return self._attributes

    @attributes.setter
    def attributes(self, val):
        self._attributes = val

    @property
    def examples(self):
        """Return self._examples."""
        return self._examples

    @examples.setter
    def examples(self, val):
        """Set self._example."""
        self._examples = val

    @property
    def attributes_size(self):
        """Return number of attributes."""
        return self._attributes.size

    @property
    def examples_size(self):
        """Return number of examples."""
        return self._examples.size

    @property
    def has_nominal_attributes(self):  # noqa
        return self._attributes.has_nominal_attributes

    def has_numeric_attributes(self):  # noqa
        return self._attributes.has_numeric_attributes

    def add(self, example=None, dataset=None, examples=None):
        """Add example or dataset to current dataset."""
        if dataset:
            self._add_dataset(dataset)
        if example:
            self._add_example(example)
        if examples:
            if isinstance(examples, list):
                for e in examples:
                    self.add(example=e)

    def _add_dataset(self, dataset):
        """Add dataset to current dataset."""
        for example in dataset.examples:
            self._examples.append(example)
        for attribute in dataset.attributes:
            self.attributes.add(attribute)

    def _add_example(self, example):
        """Add example to current dataset."""
        self._examples.append(example)

    def nominal_to_linear(self):
        """Return dataset with linear encoded attributes."""
        for i, attribute in enumerate(self.attributes):
            # Use linear encoding for bpp
            if isinstance(attribute, NominalAttribute) and len(attribute.domain) <= 2:
                continue
            mean = self.examples.mean(i)
            stdev = self.examples.stdev(i)
            for k, e in enumerate(self.examples):
                self.examples[k].values[i] = (e.values[i] - mean) / stdev

    def normalize_attributes(self):
        """Convert all attributes to binary or standardize numeric."""
        self._convert_numeric_attributes()
        self._convert_attributes_to_binary()
        self.examples.attributes = self.binary_attributes
        self._attributes = self.binary_attributes
        # print(self.binary_attributes)

    def _convert_numeric_attributes(self):
        for i, attribute in enumerate(self.attributes):
            if isinstance(attribute, NumericAttribute):
                mean = self.examples.mean(i)
                stdev = self.examples.stdev(i)
                for k, e in enumerate(self.examples):
                    self.examples[k].values[i] = float(e.values[i] - mean) / stdev

    def _convert_attributes_to_binary(self):
        # convert attributes with domain > 2
        self.binary_attributes = Attributes()
        self.new_examples = self.examples
        for i, attribute in enumerate(self.attributes):
            if isinstance(attribute, NominalAttribute) and len(attribute.domain) > 2:
                self._convert_attribute_to_binary(attribute, i)
            else:
                self.binary_attributes.add(attribute)

    def _convert_attribute_to_binary(self, attribute, attributeindex):
        """Convert one attribute to binary."""
        # conversion_dict = {}
        if len(attribute.domain) < 2:
            self.binary_attributes.add(attribute)
        else:
            for i, v in enumerate(attribute.domain):
                a = NominalAttribute(name=attribute.name + str(i))
                a.domain = [0.0, 1.0]
                a.place = i
                a.value = v
                self.binary_attributes.add(a)
                for example in self.examples:
                    if v != attribute.domain[i]:
                        example.values.insert((attributeindex + 1 + i), 0.0)
                    else:
                        example.values.insert((attributeindex + 1 + i), 1.0)
            for i, example in enumerate(self.examples):
                del example.values[attributeindex]
                self.examples._examples[i] = example
        for i, e in enumerate(self.examples):
            self.examples[i].values[attributeindex]

    # def _convert_examples_to_binary(self):
    #     """Convert examples to binary."""
    #     self.binary_examples = Examples(self.binary_attributes)
    #     for e in self.examples.to_binary(self.binary_attributes):
    #         self.binary_examples.add(e)

    ################### noqa
    #   ID3 Methods   # noqa
    ################### noqa

    def is_homogenous(self): # noqa
        """Return True if all classlabels are the same."""
        classlabel_freq = self.attribute_frequency(self.attributes.classindex)
        return len(classlabel_freq.keys()) == 1

    def get_best_split_attribute(self):
        """Return best split attribute."""
        best_gain = 0.0
        best_attributeindex = None
        for attributeindex in xrange(self.attributes.size):
            if attributeindex == self.attributes.classindex:
                continue
            current_gain = self.info_gain(attributeindex)
            if best_gain < current_gain:
                best_gain = current_gain
                best_attributeindex = attributeindex
        return best_attributeindex

    def split_on_attribute(self, attributeindex):
        """Split on best attribute."""
        splits = []
        for val in xrange(len(self.attributes[attributeindex].domain)):
            examples = [e for e in self.examples if e[attributeindex] == val]
            dataset = DataSet(examples=examples, attributes=self.attributes)
            splits.append((dataset, val))
        return splits

    def info_gain(self, attributeindex, examples=None):
        """Calculate the gained from an attribute for the class."""
        if examples is None:  # base case
            examples = self.examples
        gain = 0.0
        frequency = self.attribute_frequency(attributeindex)
        n = sum(frequency.values())
        for value, count in frequency.iteritems():
            probability = count / n
            # examples where all homogenous values for current value
            homogenous_on_attr = [e for e in examples if e[attributeindex] == value]
            # calc entropy for examples where the current attribute values
            # are similar w/ respect ot the classinde
            current_entropy = self.entropy(self.attributes.classindex, homogenous_on_attr)
            gain += probability * current_entropy
        gain = self.entropy(attributeindex, examples) - gain
        return gain

    def entropy(self, attributeindex, examples):
        """Entropy of dataset for the attribute param."""
        frequency = self.attribute_frequency(attributeindex)
        entropy = 0.0
        for val in frequency.values():
            entropy += (-val / examples.__len__()) * log(val / examples.__len__(), 2)
        return entropy

    def attribute_frequency(self, attributeindex):
        """Return frequency of attribute in data."""
        frequency = defaultdict(lambda: 0.0)
        for example in self.examples:
            frequency[example[attributeindex]] += 1.0
        return frequency

    def highest_freq_classlabel(self):
        """Return classlabel attribute value w/ the highest frequency."""
        frequency = self.attribute_frequency(self.attributes.classindex)
        classlabel = max(frequency.iteritems(), key=itemgetter(1))[0]
        return classlabel
