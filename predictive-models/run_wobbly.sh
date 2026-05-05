
CUDA_VISIBLE_DEVICES=1 python3 main_pretrain.py \
    --config-path scripts/pretrain/wobbly/ \
    --config-name qalman_knn.yaml 

CUDA_VISIBLE_DEVICES=1 python3 main_pretrain.py \
    --config-path scripts/pretrain/wobbly/ \
    --config-name qalman_kde.yaml 

CUDA_VISIBLE_DEVICES=1 python3 main_pretrain.py \
    --config-path scripts/pretrain/wobbly/ \
    --config-name qalman_logdet.yaml 
    
CUDA_VISIBLE_DEVICES=1 python3 main_pretrain.py \
    --config-path scripts/pretrain/wobbly/ \
    --config-name qalman_stopgrad.yaml 
