export OUTPUT_DIR="Experiments/QAQAFQ_CN_1"
# export CACHE_DIR=transformer_package_cache
# shellcheck disable=SC2034
CUDA_VISIBLE_DEVICES=0
python run_seq2seq.py --train_file "ConvQA/QAQAFQ/ConvQA_train.json" \
                      --output_dir ${OUTPUT_DIR} \
                      --model_type unilm \
                      --model_name_or_path "C:\pretrained_model\unilm-cn-base/" \
                      --do_lower_case \
                      --max_source_seq_length 512 \
                      --max_target_seq_length 42 \
                      --per_gpu_train_batch_size 6 \
                      --gradient_accumulation_steps 4 \
                      --learning_rate 2e-5 \
                      --num_warmup_steps 2236 \
                      --num_training_epochs 15 \
                      --save_steps 77777
                      # --cache_dir $CACHE_DIR
                      # --fp16 --fp16_opt_level O2