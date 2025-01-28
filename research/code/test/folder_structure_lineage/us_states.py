import re


statename_to_abbr = {
    # Other
    "District Of Columbia": "DC",
    "Washington Dc": "DC",
    # States
    "Alabama": "AL",
    "Montana": "MT",
    "Alaska": "AK",
    "Nebraska": "NE",
    "Arizona": "AZ",
    "Nevada": "NV",
    "Arkansas": "AR",
    "New Hampshire": "NH",
    "California": "CA",
    "New Jersey": "NJ",
    "Colorado": "CO",
    "New Mexico": "NM",
    "Connecticut": "CT",
    "New York": "NY",
    "Delaware": "DE",
    "North Carolina": "NC",
    "Florida": "FL",
    "North Dakota": "ND",
    "Georgia": "GA",
    "Ohio": "OH",
    "Hawaii": "HI",
    "Oklahoma": "OK",
    "Idaho": "ID",
    "Oregon": "OR",
    "Illinois": "IL",
    "Pennsylvania": "PA",
    "Indiana": "IN",
    "Rhode Island": "RI",
    "Iowa": "IA",
    "South Carolina": "SC",
    "Kansas": "KS",
    "South Dakota": "SD",
    "Kentucky": "KY",
    "Tennessee": "TN",
    "Louisiana": "LA",
    "Texas": "TX",
    "Maine": "ME",
    "Utah": "UT",
    "Maryland": "MD",
    "Vermont": "VT",
    "Massachusetts": "MA",
    "Virginia": "VA",
    "Michigan": "MI",
    "Washington": "WA",
    "Minnesota": "MN",
    "West Virginia": "WV",
    "Mississippi": "MS",
    "Wisconsin": "WI",
    "Missouri": "MO",
    "Wyoming": "WY",
}


def multiple_replace(lookup, text):
    regex = re.compile(r"\b(" + "|".join(lookup.keys()) + r")\b", re.IGNORECASE)

    return regex.sub(lambda mo: lookup[mo.string.title()[mo.start() : mo.end()]], text)


if __name__ == "__main__":
    text = """United States Census Regions are:
Region 1: Northeast (District of Columbia)
Division 1: New England (Connecticut, Maine, Massachusetts, New Hampshire, Rhode Island, and Vermont)
Division 2: Mid-Atlantic (New Jersey, New York, and Pennsylvania)
Region 2: Midwest (Prior to June 1984, the Midwest Region was designated as the North Central Region.)[7]
Division 3: East North Central (Illinois, Indiana, Michigan, Ohio, and Wisconsin)
Division 4: West North Central (Iowa, Kansas, Minnesota, Missouri, Nebraska, North Dakota, and South Dakota)"""

    print(multiple_replace(statename_to_abbr, text))
