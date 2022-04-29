from transformers import AutoTokenizer, T5ForConditionalGeneration, pipeline

tokenizer = AutoTokenizer.from_pretrained("checkpoint/t5-base")
model = T5ForConditionalGeneration.from_pretrained("checkpoint/t5-base")
summarizer = pipeline(task="summarization", model=model, tokenizer=tokenizer)
x = summarizer("the films written by the writer of [The Accused] starred who", max_length=30)

# use t5 in tf
# x = summarizer("An apple a day, keeps the doctor away", max_length=6, min_length=1)
print(x)