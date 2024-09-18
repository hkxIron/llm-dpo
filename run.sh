ulimit -n 64000;
<<EOF
CUDA_VISIBLE_DEVICES=0,1,2,3,4,5,6,7 python -u train.py model=dolly7b \
    datasets=[dolly] loss=dpo loss.beta=0.1 \
    exp_name=anthropic_dpo_pythia28 gradient_accumulation_steps=8 \
    batch_size=64 eval_batch_size=32 trainer=FSDPTrainer \
    sample_during_eval=false model.fsdp_policy_mp=bfloat16 \
    model.name_or_path=databricks/dolly-v2-7b \
    wandb.enabled=False
EOF

CUDA_VISIBLE_DEVICES="" python -u train.py model=tinystories \
    datasets=[stack_exchange] loss=dpo loss.beta=0.1 \
    exp_name=stackexchange_dpo_llama gradient_accumulation_steps=8 \
    batch_size=64 eval_batch_size=32 trainer=BasicTrainer \
    sample_during_eval=false model.fsdp_policy_mp=bfloat16 \
    model.name_or_path=/home/hkx/data/work/hf_data_and_model/models/TinyStories-LLaMA2-20M-256h-4l-GQA \
    wandb.enabled=False
