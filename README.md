# Food-and-Ingredient-Recognition-with-MTL
2021fall fudan computer vision course final project

- Dataset: VireoFood-172
- Model: ViT based multitask learning model
- Acc@1: 92.2%
- Acc@5: 98.8%

## Model Training
use the command: `python multitask.py --loadingre --device 1 --model vit-base-384 --batch_size 16 --batchaccum 32 --lr 1e-5 --tag 384` to train the model with the best hyperparameters.
