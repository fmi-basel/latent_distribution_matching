
#### CatProb


CUDA_VISIBLE_DEVICES=1 python3 main_pretrain.py \
    --config-path scripts/pretrain/mnist/ \
    --config-name catprob.yaml \
    ++name='"catprob_mnist_pairs_exact_p=0.992"' \
    ++method_kwargs.loss.type="exact" \
    ++method_kwargs.warmup_match_percentage=0.8 \
    ++method_kwargs.end_match_percentage=0.992 \
    ++seed=1 \
    ++max_epochs=30

CUDA_VISIBLE_DEVICES=1 python3 main_pretrain.py \
    --config-path scripts/pretrain/mnist/ \
    --config-name catprob.yaml \
    ++name='"catprob_mnist_pairs_approx2_p=0.992"' \
    ++method_kwargs.loss.type="approx" \
    ++method_kwargs.warmup_match_percentage=0.8 \
    ++method_kwargs.end_match_percentage=0.992 \
    ++method_kwargs.entropy_multiplier=2 \
    ++seed=1 \
    ++max_epochs=30 

CUDA_VISIBLE_DEVICES=1 python3 main_pretrain.py \
    --config-path scripts/pretrain/mnist/ \
    --config-name catprob.yaml \
    ++name='"catprob_mnist_pairs_approx2_p=0.8"' \
    ++method_kwargs.loss.type="approx" \
    ++method_kwargs.warmup_match_percentage=0.8 \
    ++method_kwargs.end_match_percentage=0.8 \
    ++method_kwargs.entropy_multiplier=2 \
    ++seed=1 \
    ++max_epochs=30 


CUDA_VISIBLE_DEVICES=1 python3 main_pretrain.py \
    --config-path scripts/pretrain/mnist/ \
    --config-name catprob.yaml \
    ++name='"catprob_mnist_pairs_MI"' \
    ++method_kwargs.loss.type="MI" \
    ++seed=1 \
    ++max_epochs=30 