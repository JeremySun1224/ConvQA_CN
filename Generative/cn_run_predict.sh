MODEL_PATH=E:\\Internship\\ConvQA\\Reference\\GenerativeCoQA\\Experiments\\QAQAFQ_CN\\ckpt-5105
export SPLIT=dev
export CUDA_VISIBLE_DEVICES=0
export OMP_NUM_THREADS=4
export MKL_NUM_THREADS=4
# run decoding
python decode_seq2seq.py  --model_type unilm \
                          --tokenizer_name C:/pretrained_model/unilm-cn-base \
                          --input_file "ConvQA/QAQAFQ/ConvQA_dev.json" \
                          --split $SPLIT \
                          --do_lower_case \
                          --model_path ${MODEL_PATH} \
                          --max_seq_length 512 \
                          --max_tgt_length 42 \
                          --batch_size 96 \
                          --mode s2s \
                          --output_file "ConvQA/CoQA_json/mlm_qaqafq.json" \
                          --do_rule \
                          # --beam_size 3 \
                          # --length_penalty 0 \
                          # --forbid_duplicate_ngrams \
                          # --forbid_ignore_word "." \
                          # --fp16 --amp \