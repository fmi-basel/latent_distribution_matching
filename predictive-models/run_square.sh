python3 main_pretrain.py \
    --config-path scripts/pretrain/square_motion/ \
    --config-name kalman.yaml 
    
python3 main_pretrain.py \
    --config-path scripts/pretrain/square_motion/ \
    --config-name kalman.yaml \
    ++name="square_kalmanSSL_procrustes" \
    ++method_kwargs.loss_function_type=procrustes
