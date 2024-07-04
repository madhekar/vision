from transformers import AutoTokenizer

tokenizer = AutoTokenizer.from_pretrained('roberta-large', do_lower_case=True)

example = "This is a tokenization example"

encoded = tokenizer(example)

desired_output = []
for word_id in encoded.word_ids():
    if word_id is not None:
        start, end = encoded.word_to_tokens(word_id)
        if start == end - 1:
            tokens = [start]
        else:
            tokens = [start, end-1]
        if len(desired_output) == 0 or desired_output[-1] != tokens:
            desired_output.append(tokens)
print(desired_output)