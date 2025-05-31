python finetune.py \
--data-location=./data/ \
--save=./out_lr_1/ \
--cache-dir=./cache/ \
--batch-size=32 \
--lr=1e-5

python eval_single_task.py \
--data-location=./data/ \
--save=./out_lr_1/ \
--cache-dir=./cache/ \

python eval_task_addition.py \
--data-location=./data/ \
--save=./out_lr_1/ \
--cache-dir=./cache/ \
--alpha=0.15
