import tiktoken

tokenizer = tiktoken.encoding_for_model("gpt-3.5-turbo")
text = "who is the pm of india"
tokens = tokenizer.encode(text)
print(tokens)
result_tokens = [1820, 9012, 315, 28811, 374, 1491, 35973]
result_text = tokenizer.decode(result_tokens)
print(result_text)