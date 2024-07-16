# config for training GPT-2 (124M) on a single RTX2070 (just to experiment, will take forever to train)
# python train.py config/train_gpt2_rtx2070.py

wandb_log = True
wandb_project = 'owt'
wandb_run_name='gpt2-124M'


# 1 batch size * 512 block size * 5 gradaccum = 2560 tokens per update
batch_size = 1
block_size = 512 # smaller block size for RTX2070
gradient_accumulation_steps = 5

# as the base one
max_iters = 600000
lr_decay_iters = 600000

# eval stuff
eval_interval = 1000
eval_iters = 200
log_interval = 10

# weight decay
weight_decay = 1e-1

compile = 'thunder'
