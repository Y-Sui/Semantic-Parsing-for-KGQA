#! /bin/bash

# example_all
python run.py --model_name_or_path t5-base --train_file ./data/test_339/example_all.csv --validation_file ./data/test_339/example_test.csv --text_column text --summary_column summary --output_dir output/example_all --overwrite_output_dir --do_train --do_eval --min_summ_length 10 --max_summ_length 35 --length_penalty 1.0 --per_device_train_batch_size 16 --per_device_eval_batch_size 16 --predict_with_generate --num_train_epochs 15 --max_source_length 50 --max_target_length 30 --save_steps 10000 --seed 1234

# example_train
python run.py --model_name_or_path t5-base --train_file ./data/test_339/example_train.csv --validation_file ./data/test_339/example_test.csv --text_column text --summary_column summary --output_dir output/example_train --overwrite_output_dir --do_train --do_eval --min_summ_length 10 --max_summ_length 35 --length_penalty 1.0 --per_device_train_batch_size 16 --per_device_eval_batch_size 16 --predict_with_generate --num_train_epochs 15 --max_source_length 50 --max_target_length 30 --save_steps 10000 --seed 1234

# example_finetune
python run.py --model_name_or_path t5-base --train_file ./data/test_339_finetuned/example_finetune.csv --validation_file ./data/test_339/example_test.csv --text_column text --summary_column summary --output_dir output/example_finetuned --overwrite_output_dir --do_train --do_eval --min_summ_length 10 --max_summ_length 35 --length_penalty 1.0 --per_device_train_batch_size 64 --per_device_eval_batch_size 64 --predict_with_generate --num_train_epochs 3 --max_source_length 50 --max_target_length 30 --save_steps 10000 --seed 1234

set -e
# 压缩包名称
zip -q -r output.zip ./output
# 通过oss上传到个人数据中的backup文件夹中
oss cp output.zip oss://backup/
rm -f output.zip

# 传输成功后关机
shutdown
