@echo off
echo S-Diff Quick Test
echo.

cd /d "C:/Users/mog/Desktop/机器学习/论文/ReChorus\src"

echo Test 1: Paper settings (guidance_s=0.02)
python main.py --model_name SDiff --dataset ml-1m ^
    --T 5 --alpha_min 0.01 --sigma_max 0.5 --K_eig 200 ^
    --lr 0.0001 --batch_size 128 --emb_size 256 --guidance_s 0.02 ^
    --epoch 20 --test_epoch 4 --early_stop 4 --device cpu

echo.
echo ========================================
echo.

echo Test 2: No CFG (guidance_s=0.0)
python main.py --model_name SDiff --dataset ml-1m ^
    --T 5 --alpha_min 0.01 --sigma_max 0.5 --K_eig 200 ^
    --lr 0.0001 --batch_size 128 --emb_size 256 --guidance_s 0.0 ^
    --epoch 20 --test_epoch 4 --early_stop 4 --device cpu

echo.
echo ========================================
echo.

echo Test 3: Strong CFG (guidance_s=0.1)
python main.py --model_name SDiff --dataset ml-1m ^
    --T 5 --alpha_min 0.01 --sigma_max 0.5 --K_eig 200 ^
    --lr 0.0001 --batch_size 128 --emb_size 256 --guidance_s 0.1 ^
    --epoch 20 --test_epoch 4 --early_stop 4 --device cpu

pause
