@echo off
echo S-Diff Single Experiment
echo.

cd /d "C:/Users/mog/Desktop/机器学习/论文/ReChorus\src"

set T=5
set ALPHA_MIN=0.01
set SIGMA_MAX=0.5
set K_EIG=200
set LR=0.0001
set BATCH_SIZE=128
set EMB_SIZE=256
set GUIDANCE_S=0.02

echo Running with parameters:
echo   T=%%T%%
echo   alpha_min=%%ALPHA_MIN%%
echo   sigma_max=%%SIGMA_MAX%%
echo   K_eig=%%K_EIG%%
echo   lr=%%LR%%
echo   batch_size=%%BATCH_SIZE%%
echo   emb_size=%%EMB_SIZE%%
echo   guidance_s=%%GUIDANCE_S%%
echo.

python main.py --model_name SDiff --dataset ml-1m ^
    --T %%T%% ^
    --alpha_min %%ALPHA_MIN%% ^
    --sigma_max %%SIGMA_MAX%% ^
    --K_eig %%K_EIG%% ^
    --lr %%LR%% ^
    --batch_size %%BATCH_SIZE%% ^
    --emb_size %%EMB_SIZE%% ^
    --guidance_s %%GUIDANCE_S%% ^
    --epoch 30 ^
    --test_epoch 5 ^
    --early_stop 5 ^
    --l2 0 ^
    --optimizer Adam ^
    --metric NDCG,HR ^
    --topk 5,10 ^
    --device cpu

pause
