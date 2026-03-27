from transformers import pipeline

def summarize_text_ai(text, max_length=150, min_length=40):
    # Load a pre-trained summarization model (e.g., "t5-small")
    summarizer = pipeline("summarization", model="t5-small")
    
    # Generate the summary
    summary = summarizer(text, max_length=max_length, min_length=min_length, do_sample=False)
    
    return summary[0]['summary_text']

# Example usage with a long text
long_text = """[Insert your long text here, e.g., a news article or research paper snippet]"""

# print(summarize_text_ai(long_text))
