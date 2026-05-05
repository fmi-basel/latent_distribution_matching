

#### Gaussprob


CUDA_VISIBLE_DEVICES=$2 python3 main_pretrain.py \
    --config-path scripts/pretrain/cifar/ \
    --config-name gaussprob.yaml \
    ++name='"gaussprob_cifar10_dual_sample"' \
    ++method_kwargs.loss.type="dual_sample" \
    ++seed=$1 \
    ++max_epochs=1000 

CUDA_VISIBLE_DEVICES=$2 python3 main_pretrain.py \
    --config-path scripts/pretrain/cifar/ \
    --config-name gaussprob.yaml \
    ++name='"gaussprob_cifar10_dual_knn"' \
    ++method_kwargs.loss.type="dual_knn" \
    ++method_kwargs.prediction_precision=12.5 \
    ++seed=$1 \
    ++max_epochs=1000 

CUDA_VISIBLE_DEVICES=$2 python3 main_pretrain.py \
    --config-path scripts/pretrain/cifar/ \
    --config-name gaussprob.yaml \
    ++name='"gaussprob_cifar10_single_sample"' \
    ++method_kwargs.loss.type="single_sample" \
    ++seed=$1 \
    ++max_epochs=1000 

CUDA_VISIBLE_DEVICES=$2 python3 main_pretrain.py \
    --config-path scripts/pretrain/cifar/ \
    --config-name gaussprob.yaml \
    ++name='"gaussprob_cifar10_single_knn"' \
    ++method_kwargs.loss.type="single_knn" \
    ++method_kwargs.prediction_precision=12.5 \
    ++seed=$1 \
    ++max_epochs=1000 

## VicREG

CUDA_VISIBLE_DEVICES=$2 python3 main_pretrain.py \
    --config-path scripts/pretrain/cifar/ \
    --config-name vicreg.yaml \
    ++seed=$1 \
    ++max_epochs=1000 

CUDA_VISIBLE_DEVICES=$2 python3 main_pretrain.py \
    --config-path scripts/pretrain/cifar/ \
    --config-name vicreg2.yaml \
    ++name='"vicreg2_cifar10_lower_entropy"' \
    ++data.dataset='"cifar10"' \
    ++method_kwargs.var_loss_weight=17.5 \
    ++method_kwargs.cov_loss_weight=0.75 \
    ++seed=$1 \
    ++max_epochs=1000 


#### SphereProb


CUDA_VISIBLE_DEVICES=$2 python3 main_pretrain.py \
    --config-path scripts/pretrain/cifar/ \
    --config-name sphereprob.yaml \
    ++name='"sphereprob_cifar10_dual_sample"' \
    ++method_kwargs.loss.type="dual_sample" \
    ++seed=$1 \
    ++max_epochs=1000 

CUDA_VISIBLE_DEVICES=$2 python3 main_pretrain.py \
    --config-path scripts/pretrain/cifar/ \
    --config-name sphereprob.yaml \
    ++name='"sphereprob_cifar10_dual_gauss"' \
    ++method_kwargs.loss.type="dual_gauss" \
    ++seed=$1 \
    ++max_epochs=1000 

CUDA_VISIBLE_DEVICES=$2 python3 main_pretrain.py \
    --config-path scripts/pretrain/cifar/ \
    --config-name sphereprob.yaml \
    ++name='"sphereprob_cifar10_dual_knn"' \
    ++method_kwargs.loss.type="dual_knn" \
    ++seed=$1 \
    ++max_epochs=1000 ; 
CUDA_VISIBLE_DEVICES=$2 python3 main_pretrain.py \
    --config-path scripts/pretrain/cifar/ \
    --config-name sphereprob.yaml \
    ++name='"sphereprob_cifar10_single_sample"' \
    ++data.dataset='"cifar10"' \
    ++method_kwargs.loss.type="single_sample" \
    ++seed=$1 \
    ++max_epochs=1000 ; 
CUDA_VISIBLE_DEVICES=$2 python3 main_pretrain.py \
    --config-path scripts/pretrain/cifar/ \
    --config-name sphereprob.yaml \
    ++name='"sphereprob_cifar10_single_gauss"' \
    ++data.dataset='"cifar10"' \
    ++method_kwargs.loss.type="single_gauss" \
    ++seed=$1 \
    ++max_epochs=1000 ; 
CUDA_VISIBLE_DEVICES=$2 python3 main_pretrain.py \
    --config-path scripts/pretrain/cifar/ \
    --config-name sphereprob.yaml \
    ++name='"sphereprob_cifar10_single_knn"' \
    ++data.dataset='"cifar10"' \
    ++method_kwargs.loss.type="single_knn" \
    ++seed=$1 \
    ++max_epochs=1000 