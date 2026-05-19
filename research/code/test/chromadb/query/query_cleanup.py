import re

def _preprocess_query(query: str) -> str:
    query = query.strip().lower()
    query = re.sub(r'[^\w\s?!]', '', query)  # Strip special characters
    return " ".join(query.split())  # Remove extra spaces