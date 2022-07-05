#! /bin/bash

# Sample of using t5-base, before this cloning the t5-base from the huggingface to ./checkpoint
# 1. pre-finetuning using Metaqa datasets
python run.py --model_name_or_path checkpoint/t5-base --train_file ./data/train_csv_MetaQA.csv --validation_file ./data/eval_csv_MetaQA.csv --text_column text --summary_column summary --output_dir checkpoint/t5-base --overwrite_output_dir --do_train --min_summ_length 10 --max_summ_length 40 --length_penalty 1.0 --per_device_train_batch_size 8 --per_device_eval_batch_size 8 --predict_with_generate --num_train_epochs 15 --max_source_length 50 --max_target_length 30 --save_steps 10000 --seed 1234

# 2. finetuning on example_339 datasets (using both test and train sets), the output will be saved at ./output/example_finetuned_post
python run.py --model_name_or_path checkpoint/t5-base --train_file ./data/test_339/example_all.csv --validation_file ./data/test_339/example_test.csv --text_column text --summary_column summary --output_dir output/example_finetuned_post --do_train --do_eval --overwrite_output_dir  --min_summ_length 10 --max_summ_length 40 --length_penalty 0.8 --per_device_train_batch_size 4 --per_device_eval_batch_size 4 --predict_with_generate --num_train_epochs 50 --max_source_length 50 --max_target_length 30 --save_steps 10000 --seed 1234

set -e
# 3. zip the ./output and upload to the OSS
zip -q -r output.zip ./output
oss cp output.zip oss://backup/
rm -f output.zip

# 4. shutdown
shutdown
