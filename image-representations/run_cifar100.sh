

#### Gaussprob


CUDA_VISIBLE_DEVICES=$2 python3 main_pretrain.py \
    --config-path scripts/pretrain/cifar/ \
    --config-name gaussprob.yaml \
    ++name='"gaussprob_cifar100_dual_sample"' \
    ++data.dataset='"cifar100"' \
    ++method_kwargs.loss.type="dual_sample" \
    ++max_epochs=1000 \
    ++seed=$1 

CUDA_VISIBLE_DEVICES=$2 python3 main_pretrain.py \
    --config-path scripts/pretrain/cifar/ \
    --config-name gaussprob.yaml \
    ++name='"gaussprob_cifar100_dual_knn"' \
    ++data.dataset='"cifar100"' \
    ++method_kwargs.loss.type="dual_knn" \
    ++max_epochs=1000 \
    ++seed=$1 

CUDA_VISIBLE_DEVICES=$2 python3 main_pretrain.py \
    --config-path scripts/pretrain/cifar/ \
    --config-name gaussprob.yaml \
    ++name='"gaussprob_cifar100_single_sample"' \
    ++data.dataset='"cifar100"' \
    ++method_kwargs.loss.type="single_sample" \
    ++max_epochs=1000 \
    ++seed=$1 

CUDA_VISIBLE_DEVICES=$2 python3 main_pretrain.py \
    --config-path scripts/pretrain/cifar/ \
    --config-name gaussprob.yaml \
    ++name='"gaussprob_cifar100_single_knn"' \
    ++data.dataset='"cifar100"' \
    ++method_kwargs.loss.type="single_knn" \
    ++max_epochs=1000 \
    ++seed=$1 

## VicREG

CUDA_VISIBLE_DEVICES=$2 python3 main_pretrain.py \
    --config-path scripts/pretrain/cifar/ \
    --config-name vicreg.yaml \
    ++name='"vicreg-cifar100"' \
    ++data.dataset='"cifar100"' \
    ++max_epochs=1000 \
    ++seed=$1 

CUDA_VISIBLE_DEVICES=$2 python3 main_pretrain.py \
    --config-path scripts/pretrain/cifar/ \
    --config-name vicreg2.yaml \
    ++name='"vicreg2_cifar100_lower_entropy"' \
    ++data.dataset='"cifar100"' \
    ++method_kwargs.var_loss_weight=17.5 \
    ++method_kwargs.cov_loss_weight=0.75 \
    ++max_epochs=1000 \
    ++seed=$1 


#### SphereProb

CUDA_VISIBLE_DEVICES=$2 python3 main_pretrain.py \
    --config-path scripts/pretrain/cifar/ \
    --config-name sphereprob.yaml \
    ++name='"sphereprob_cifar100_dual_sample_prc=2.5"' \
    ++data.dataset='"cifar100"' \
    ++method_kwargs.loss.type="dual_sample" \
    ++method_kwargs.prediction_precision=2.5 \
    ++max_epochs=1000 \
    ++seed=$1 

CUDA_VISIBLE_DEVICES=$2 python3 main_pretrain.py \
    --config-path scripts/pretrain/cifar/ \
    --config-name sphereprob.yaml \
    ++name='"sphereprob_cifar100_dual_gauss_prc=2.5"' \
    ++data.dataset='"cifar100"' \
    ++method_kwargs.loss.type="dual_gauss" \
    ++method_kwargs.prediction_precision=2.5 \
    ++max_epochs=1000 \
    ++seed=$1 

CUDA_VISIBLE_DEVICES=$2 python3 main_pretrain.py \
    --config-path scripts/pretrain/cifar/ \
    --config-name sphereprob.yaml \
    ++name='"sphereprob_cifar100_dual_knn_prc=2.5"' \
    ++data.dataset='"cifar100"' \
    ++method_kwargs.loss.type="dual_knn" \
    ++method_kwargs.prediction_precision=2.5 \
    ++max_epochs=1000 ; \
    ++seed=$1 

CUDA_VISIBLE_DEVICES=$2 python3 main_pretrain.py \
    --config-path scripts/pretrain/cifar/ \
    --config-name sphereprob.yaml \
    ++name='"sphereprob_cifar100_single_sample"' \
    ++data.dataset='"cifar100"' \
    ++method_kwargs.loss.type="single_sample" \
    ++max_epochs=1000 ; \
    ++seed=$1
    
CUDA_VISIBLE_DEVICES=$2 python3 main_pretrain.py \
    --config-path scripts/pretrain/cifar/ \
    --config-name sphereprob.yaml \
    ++name='"sphereprob_cifar100_single_gauss"' \
    ++data.dataset='"cifar100"' \
    ++method_kwargs.loss.type="single_gauss" \
    ++max_epochs=1000 ; \
    ++seed=$1 

CUDA_VISIBLE_DEVICES=$2 python3 main_pretrain.py \
    --config-path scripts/pretrain/cifar/ \
    --config-name sphereprob.yaml \
    ++name='"sphereprob_cifar100_single_knn"' \
    ++data.dataset='"cifar100"' \
    ++method_kwargs.loss.type="single_knn" \
    ++max_epochs=1000 
    ++seed=$1 