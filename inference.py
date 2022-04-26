from transformers import AutoTokenizer, T5ForConditionalGeneration, pipeline
import pandas as pd
import numpy as np
import torch

tokenizer = AutoTokenizer.from_pretrained("checkpoint/t5-base")
model = T5ForConditionalGeneration.from_pretrained("checkpoint/t5-base")
summarizer = pipeline(task="summarization", model=model, tokenizer=tokenizer)

# input_query = ["what are the languages spoken in the movies whose directors also directed [Son of Dracula]",
#          "what are the languages spoken in the movies whose directors also directed [Son of Dracula]"]
input_query_num = 50
input_query = pd.read_csv("./data/eval_csv_MetaQA.csv")["text"]
input_query = input_query[:input_query_num].to_list()
golden_path_target = pd.read_csv("./data/eval_csv_MetaQA.csv")["summary"]
golden_path_target = golden_path_target[:input_query_num].to_list()

# obtain the embedding representation of the path targets
df = pd.read_csv("./data/train_csv.csv")
df = df.drop_duplicates(subset=['summary'])
path_targets = df["summary"].to_list()
path_tokenzier = tokenizer(
    path_targets,
    return_tensors="pt",
    padding=True,
    truncation=True
)
path_target_embedding = model.encoder(
    input_ids=path_tokenzier["input_ids"],
    attention_mask=path_tokenzier["attention_mask"],
    return_dict=True
)
path_target_embedding = path_target_embedding.last_hidden_state

# obtain the representation of the output of the query
output_query = summarizer(input_query, max_length=20, min_length=18)
output_query_list = []
for i in range(len(output_query)):
    output_query_list.append(output_query[i]["summary_text"])
output_query_tokenzier = tokenizer(
    output_query_list,
    return_tensors="pt",
    padding=True,
    truncation=True
)
output_query_embedding = model.encoder(
    input_ids=output_query_tokenzier["input_ids"],
    attention_mask=output_query_tokenzier["attention_mask"],
    return_dict=True
)
output_query_embedding = output_query_embedding.last_hidden_state

# calculate the vector similarity
def vector_similarity(v1, v2):
    # define the sentence vector
    def sentence_vector(v):
        sentence_vector = 0.0
        for i in range(v.shape[0]):
            sentence_vector += v[i]
        return sentence_vector / v.shape[0]
    v1, v2 = sentence_vector(v1), sentence_vector(v2)
    return torch.cosine_similarity(v1, v2, dim=0, eps=1e-6)

# replace the output with the high similarity target path
transfer_output = []
for i in range(len(output_query_list)):
    print(f"{output_query[i]['summary_text']}")
    target_path_index = 0
    max_score = 0.0
    for j in range(len(path_targets)):
        score = vector_similarity(output_query_embedding[i], path_target_embedding[j])
        print(f"{path_targets[j]} score={score:.6f}")
        if score > max_score:
            max_score = score
            target_path_index = j
        else:
            max_score = max_score
    print(f"Results: {output_query[i]['summary_text']}-> {path_targets[target_path_index]} with confidence {max_score:.6f}")
    transfer_output.append(path_targets[target_path_index])

# evaluate the t5 seq2seq
assert len(transfer_output) == len(golden_path_target)
acc = 0
for i in range(len(transfer_output)):
    if transfer_output[i] == golden_path_target[i]:
        acc += 1
print(f"Accuracy: {acc/len(transfer_output)}")