import re

def are_strings_equal_by_tokens(s1, s2):
    """
    Determines if two strings are equal based on their unique, case-insensitive word tokens.
    Punctuation is removed before tokenization.
    """
    # Normalize strings: lowercase and remove non-alphanumeric characters
    s1_normalized = re.sub(r'[^\w\s]', '', s1).lower()
    s2_normalized = re.sub(r'[^\w\s]', '', s2).lower()

    # Tokenize into words
    tokens1 = set(s1_normalized.split())
    tokens2 = set(s2_normalized.split())

    # Compare the sets of tokens
    return tokens1 == tokens2

# Example Usage:
string1 = "Anjali, Kumar and a person"
string2 = "Kumar, Anjali and a person"
string3 = "This is a different test."

print(f"'{string1}' and '{string2}' are equal by tokens: {are_strings_equal_by_tokens(string1, string2)}")
print(f"'{string1}' and '{string3}' are equal by tokens: {are_strings_equal_by_tokens(string1, string3)}")

string4 = "Apple Banana Orange"
string5 = "Banana Orange Apple"
print(f"'{string4}' and '{string5}' are equal by tokens: {are_strings_equal_by_tokens(string4, string5)}")