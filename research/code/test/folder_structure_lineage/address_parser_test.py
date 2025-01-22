import pyap
test_address = """
    13580 sage mesa rd, san diego, CA 92130
    """
addresses = pyap.parse(test_address, country='US')
for address in addresses:
        # shows found address
        #print(address)
        # shows address parts
        print(address.as_dict())
