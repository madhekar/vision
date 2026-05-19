import re

def _preprocess_query(query: str) -> str:
    query = query.strip().lower()
    query = re.sub(r'[^\w\s?!]', '', query)  # Strip special characters
    return " ".join(query.split())  # Remove extra spaces


# examples

q1 = "Esha -  performing Bharathnatyam dance."
q2 = "Alaska Glauysers in   seaward."


print(f"original: {q1} processed: {_preprocess_query(q1)}")
print(f"original: {q2} processed: {_preprocess_query(q2)}")