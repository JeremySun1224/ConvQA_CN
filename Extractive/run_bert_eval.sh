python run_coqa.py --model_type bert \
                   --model_name_or_path "E:\Internship\ConvQA\Reference\transformers-coqa\bert-output-rtransformer/" \
                   --do_eval \
                   --data_dir "E:\Internship\ConvQA\Reference\transformers-coqa\data\raw_data_cn_v3_sub" \
                   --train_file ConvQA_CN_v3.0_train.json \
                   --predict_file ConvQA_CN_v3.0_dev.json \
                   --max_seq_length 512 \
                   --doc_stride 384 \
                   --learning_rate 3e-5 \
                   --num_train_epochs 5 \
                   --output_dir bert-output-rtransformer/ \
                   --do_lower_case \
                   --per_gpu_train_batch_size 32  \
                   --per_gpu_eval_batch_size 64 \
                   --gradient_accumulation_steps 1 \
                   --max_grad_norm -1 \
                   --threads 6 \
                   --weight_decay 0.01