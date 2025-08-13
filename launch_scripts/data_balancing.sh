python finetune.py \
--data-location=./data/ \
--save=./out_db/ \
--cache-dir=./cache/ \
--batch-size=32 \
--lr=1e-4 \
--wd=0.001 \
--undersample

python eval_single_task.py \
--data-location=./data/ \
--save=./out_db/ \
--cache-dir=./cache/

python eval_task_addition.py \
--data-location=./data/ \
--save=./out_db/ \
--cache-dir=./cache/
