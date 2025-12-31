
#!/bin/bash
# Quick grid search for S-Diff

DATASET="ml-1m"
BASE_CMD="python main.py --model_name SDiff --epoch 30 --test_epoch 5 --early_stop 3"

# 只测试关键参数
for T in 5 10
do
    for lr in 1e-4 5e-4
    do
        for emb_size in 128 256
        do
            for guidance_s in 0.0 0.02 0.2
            do
                CMD="$BASE_CMD --dataset $DATASET --T $T --lr $lr --emb_size $emb_size --guidance_s $guidance_s"
                echo "Running: $CMD"
                eval $CMD
                echo "----------------------------------------"
            done
        done
    done
done
