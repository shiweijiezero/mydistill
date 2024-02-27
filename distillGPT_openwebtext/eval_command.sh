lm_eval --model hf \
    --model_args pretrained=./save_model/gpt2_hf_end,trust_remote_code=true,dtype=bfloat16 \
    --tasks glue,gsm8k,boolq,mmlu,cmmlu  \
    --batch_size 64 \
    --output_path ./eval_out/gpt2_openwebtext \
    --device cuda:0