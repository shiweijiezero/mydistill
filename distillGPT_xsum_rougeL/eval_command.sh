lm_eval --model hf \
    --model_args pretrained=./save_model/gpt2_hf_end,dtype=bfloat16 \
    --tasks glue,gsm8k,boolq,mmlu,cmmlu  \
    --batch_size 64 \
    --output_path ./eval_out/gpt2_xsum \
    --device cuda:4