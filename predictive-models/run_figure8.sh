

python3 main_pretrain.py \
    --config-path scripts/pretrain/figure8/ \
    --config-name kalman.yaml 


CUDA_VISIBLE_DEVICES=1 python3 main_pretrain.py \
    --config-path scripts/pretrain/figure8/ \
    --config-name kalman.yaml \
    ++name="kalmanSSL_figure8_procrustes" \
    ++method_kwargs.loss_function_type=procrustes
