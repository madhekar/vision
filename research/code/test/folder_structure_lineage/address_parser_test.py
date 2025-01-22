import pyap as ap

def parse_address(full_add):
    parse_address = ap.parse(full_add, country="US")
    for add in parse_address:
        print(add.as_dict())

test_address = "13580 sage mesa rd, san diego, CA 92130"
parse_address(test_address)
# addresses = ap.parse(test_address, country='US')
# print(addresses)
# for address in addresses:
#         print(address.as_dict())
