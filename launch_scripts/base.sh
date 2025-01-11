# 1) finetune on all datasets
# 2) eval single task
# 3) task addition

python finetune.py \
--data-location=./data/ \
--save=./out/ \
--batch-size=32 \
--lr=1e-4 \
--wd=0.0

python eval_single_task.py \
--data-location=./data/ \
--save=./out/

python eval_task_addition.py \
--data-location=./data/ \
--save=./out/

exec bash