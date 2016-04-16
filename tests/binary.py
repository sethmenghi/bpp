    # def _convert_attributes_to_binary(self):
    #     # convert attributes with domain > 2
    #     self.binary_attributes = Attributes()
    #     for i, attribute in enumerate(self.attributes):
    #         if isinstance(attribute, NominalAttribute) and len(attribute.domain) > 2:
    #             self._convert_attribute_to_binary(attribute, i)
    #         else:
    #             self.binary_attributes.add(attribute)

    # def _convert_attribute_to_binary(self, attribute, attributeindex):
    #     """Convert one attribute to binary."""
    #     conversion_dict = {}
    #     if len(attribute.domain) < 2:
    #         self.binary_attributes.add(attribute)
    #     else:
    #         for i, v in enumerate(attribute.domain):
    #             binaryrepr = ('0' * (len(attribute.domain)))
    #             attribute = BinaryAttribute(name=attribute.name)
    #             attribute.domain = [0, 1]
    #             attribute.place = i
    #             attribute.value = v
    #             list(binaryrepr)[i] = '1'
    #             conversion_dict[v] = ''.join(binaryrepr)
    #             self.binary_attributes.add(attribute)
    #     for i, e in enumerate(self.examples):
    #         print("%s --> %s" % (e.values[attributeindex], conversion_dict[e.values[attributeindex]]))
    #         e.values[attributeindex] = conversion_dict[e.values[attributeindex]]
    #         self.examples[i] = e

    # def _convert_examples_to_binary(self):
    #     """Convert examples to binary."""
    #     self.binary_examples = Examples(self.binary_attributes)
    #     for e in self.examples.to_binary(self.binary_attributes):
    #         self.binary_examples.add(e)