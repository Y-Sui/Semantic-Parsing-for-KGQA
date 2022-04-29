# Semantic Parsing for KGQA

## Baseline
- STAGG is one of the most foundation work of semantic parsing in knowledge graph based question answering. However, due to the old version of its components: **Entity linking (EL) and Inferential Chain Prediction**, it gives unpromising performance.
- One intuition to solve this problem is to update the blocks of the whole model, like using transformers-based model as compensation.

![](figures/Snipaste_2022-04-29_14-14-06.png)

## Mention Detection & Entity Linking
- Entity Linking can be separated from the KGQA as a specific task.
- From **S-MART** to **BERT-Joint Training**.

![](figures/图片1.png)
References:S-MART: Novel Tree-based Structured Learning Algorithms Applied to Tweet Entity Linking: https://arxiv.org/abs/1609.08075

### Ablation Study
![](figures/图片2.png)
From Table 1, we have two observations:
The overall F1 of the whole baseline is 56.45%, which is unpromising but reasonable. Because the topic entity linking module obtains 72.64% F1, and the inferential chain prediction module only achieves 65.88% F1.
Once simply replace the entity linking module from SMART-ACL (the module EL used in the baseline) to BERT + Joint training, it obtains almost **14% boosting** on this important step.

## Core Inferential Chain Generation

Inferential chain prediction involving two steps: Staged Query Graph Generation and inferential chain prediction

The main objective of this module is to fine-tune and evaluate a model (pre-trained on a large-scale dataset) on domain-specical task. Here, to construct the core inferential chain generation, I have taken an example of finetuning sequence-to-sequence models such as T5, on an abstractive summarization task using the Trainer API from Hugging Face.

* A number of pre-trained models can be finetuned such as:
    * T5 (small, base, large, 3B, 11B)
    * BART (base, large-cnn, large-mnli)
    * Longformer Encoder Decoder (allenai/led-base-16384, allenai/led-large-16384)
    * Pegasus (large, xsum, multi_news)

Checkout [pre-trained models](https://huggingface.co/models) to see the checkpoints available for each of them.

### Datasets
Finetuning with MetaQA dataset placed at [`data`](https://github.com/Yuan-BertSui/seq2seq/tree/master/data).

Here's an sample of the training dataset, it has two columns, one is the text, and other one is the summary of the text (actually it is the inferential chain of the original text):
| text                                                                                    | summary                              |
|-----------------------------------------------------------------------------------------|--------------------------------------|
| The films that share actors with the [Dil Chahta Hai] film have been released for years | movie_to_actor_to_movie_to_year      |
| which are the directors of the films written by the writer of [The Green Mile]          | movie_to_writer_to_movie_to_director |

Tips: [] is the entity mention of the query sentences. To be noted that there's no need to label the entity mention at this stage.

### Script
Run the pretrained model using the following script.

```bash
python run.py \
    --model_name_or_path t5-base \
    --train_file ./data/train_csv_MetaQA.csv \
    --validation_file ./data/eval_csv_MetaQA.csv \
    --text_column text \
    --summary_column summary \
    --output_dir output/ \
    --overwrite_output_dir \
    --do_train \
    --do_eval \
    --num_beams 3 \
    --min_summ_length 5 \
    --max_summ_length 20 \
    --length_penalty 1.0 \
    --per_device_train_batch_size 8 \
    --per_device_eval_batch_size 8 \
    --predict_with_generate \
    --num_train_epochs 5 \
    --fp16 \
    --max_source_length 25 \
    --max_target_length 20 \
    --save_steps 10000 \
    --seed 1234 \
```
The `model_name_or_path' refers to the pretrained_model files' name, ``train_file' refers to the training dataset of the model, 'validation_file' refers to the validation dataset of the model, 'text_column' refers to the query which needs to obtain the correponding core inferential chain, 'summary_column' refers to the corresponding core inferential of the query.
To be noted that the 'per_device_train/eval_batch_size' have a huge impact on the performance of the finetuning.

To see all the possible command line options, run:

```bash
python run.py --help
```

### Inference
Since the t5 is a generative model, the form of the generated sequence is uncertain. It needs to match the core inferential chain in our dataset. One simple intuition is to calculate the cosine similarity of both the generated sequence and the core inferential chain with the help of the embedding representation of the t5 model.

To see the inference model, please run the follow command:
```bash
python inference.py
```





