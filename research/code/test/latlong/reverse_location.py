# 1. Full formatted address string
print(location.address)
# Output: "San Diego City Hall, 202, C Street, Civic Core, Downtown, San Diego, San Diego County, California, 92101, United States"

# 2. Structured address components (Nominatim format)
address_parts = location.raw.get('address', {})

street = f"{address_parts.get('house_number', '')} {address_parts.get('road', '')}".strip()
city = address_parts.get('city') or address_parts.get('town') or address_parts.get('village')
state = address_parts.get('state')
country = address_parts.get('country')

print(f"{street} {city}, {state} {country}")
# Output: "202 C Street | San Diego, California 92101"
