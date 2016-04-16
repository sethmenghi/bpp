"""DT holds the DecisionTree object."""

from classifier import Classifier


class Node(object):
    """Node of a decision tree."""

    def __init__(self, attribute=None, classlabel=None, is_leaf=False):
        """Put attribute for node along with classlabel if leaf node."""
        self._children = []
        self._attribute = attribute
        self.attribute_val = None
        self._is_leaf = is_leaf
        self.classlabel = classlabel

    def __getitem__(self, val):
        """Allow for indexing."""
        for child in self.children:
            if child.attribute_val == val:
                return child

    @property
    def children(self):
        """Return array of children."""
        return self._children

    @property
    def attribute(self):
        """Return attribute of Node."""
        return self._attribute

    @property
    def is_leaf(self):
        """Return True if leaf node."""
        return self._is_leaf

    def append(self, child):
        """Append a node to children."""
        return self.children.append(child)

    def __repr__(self):
        """Representation of a classlabel."""
        string = "%s\t%s" % (str(self.attribute), str(self.attribute_val))
        return string


class DecisionTree(Classifier):
    """ID3 decision tree classification model."""

    model = "ID3"

    def classify(self, example):
        """Classify."""
        classification = self._classify(example, self.root)
        return classification

    def _classify(self, example, node):
        if node.is_leaf:
            return node.classlabel
        if node.attribute is None:
            return node.classlabel
        child_node = node[example[node.attribute]]
        return self._classify(example, child_node)

    def train(self, example):
        """Run training, needs to be overloaded for recursion."""
        self.root = self._train(self.trainset)

    def _train(self, dataset):
        """Recursive train create."""
        if dataset.is_homogenous() or dataset.attributes_size <= 1:
            highest_frequency_class = dataset.highest_freq_classlabel()
            return Node(classlabel=highest_frequency_class, is_leaf=True)
        elif dataset.examples_size == 0 or dataset.attributes_size == 0 or dataset is None:
            return
        else:
            node = Node(attribute=dataset.get_best_split_attribute(), is_leaf=False)
            splits = dataset.split_on_attribute(node.attribute)
            for split, attribute_val in splits:
                current_child = self._train(split)
                if current_child:
                    current_child.attribute_val = attribute_val
                    node.append(current_child)
                else:
                    node._is_leaf = True
            highest_frequency_class = dataset.highest_freq_classlabel()
            node.classlabel = highest_frequency_class
            # The root is saved to keep _classify consistent
            return node
