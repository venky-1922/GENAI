import tiktoken

tokenizer = tiktoken.encoding_for_model("gpt-3.5-turbo")
text = "who is the pm of india"
tokens = tokenizer.encode(text)
print(tokens)
