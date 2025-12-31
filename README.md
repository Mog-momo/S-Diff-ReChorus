# S-Diff-ReChorus

A diffusion-based sequential recommendation framework evaluated on the **Grocery_and_Gourmet_Food** dataset from Amazon Reviews.

## Supported Models & Commands

Run any of the following models using the unified entry point:

```bash
# Main model (with FiLM)
python src/main.py --model_name SDiff --dataset Grocery_and_Gourmet_Food --num_workers 0

# Ablation: without FiLM
python src/main.py --model_name SDiff_w_o_FiLM --dataset Grocery_and_Gourmet_Food --num_workers 0

# Ablation: standard DDPM in spectral domain
python src/main.py --model_name DDPM_in_Spectral --dataset Grocery_and_Gourmet_Food --num_workers 0