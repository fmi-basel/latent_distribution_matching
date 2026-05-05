

#### Gaussprob

python3 main_pretrain.py \
    --config-path scripts/pretrain/imagenet-100/ \
    --config-name vicreg.yaml \
    ++name='"vicreg-imagenet100"' \
    ++seed=$1 

python3 main_pretrain.py \
    --config-path scripts/pretrain/imagenet-100/ \
    --config-name vicreg2.yaml \
    ++name='"vicreg2_imagenet100_lower_entropy"' \
    ++method_kwargs.var_loss_weight=17.5 \
    ++method_kwargs.cov_loss_weight=0.75 \
    ++seed=$1 

python3 main_pretrain.py \
    --config-path scripts/pretrain/imagenet-100/ \
    --config-name gaussprob.yaml \
    ++name='"gaussprob_imagenet100_dual_sample"' \
    ++method_kwargs.loss.type="dual_sample" \
    ++seed=$1 

python3 main_pretrain.py \
    --config-path scripts/pretrain/imagenet-100/ \
    --config-name gaussprob.yaml \
    ++name='"gaussprob_imagenet100_dual_knn"' \
    ++method_kwargs.loss.type="dual_knn" \
    ++seed=$1 

python3 main_pretrain.py \
    --config-path scripts/pretrain/imagenet-100/ \
    --config-name gaussprob.yaml \
    ++name='"gaussprob_imagenet100_single_sample"' \
    ++method_kwargs.loss.type="single_sample" \
    ++seed=$1 

python3 main_pretrain.py \
    --config-path scripts/pretrain/imagenet-100/ \
    --config-name gaussprob.yaml \
    ++name='"gaussprob_imagenet100_single_knn"' \
    ++method_kwargs.loss.type="single_knn" \
    ++seed=$1 


#### SphereProb

python3 main_pretrain.py \
    --config-path scripts/pretrain/imagenet-100/ \
    --config-name sphereprob.yaml \
    ++name='"sphereprob_imagenet100_dual_sample_prc=2.5"' \
    ++method_kwargs.loss.type="dual_sample" \
    ++method_kwargs.prediction_precision=2.5 \
    ++seed=$1 

python3 main_pretrain.py \
    --config-path scripts/pretrain/imagenet-100/ \
    --config-name sphereprob.yaml \
    ++name='"sphereprob_imagenet100_dual_gauss_prc=2.5"' \
    ++method_kwargs.loss.type="dual_gauss" \
    ++method_kwargs.prediction_precision=2.5 \
    ++seed=$1 

python3 main_pretrain.py \
    --config-path scripts/pretrain/imagenet-100/ \
    --config-name sphereprob.yaml \
    ++name='"sphereprob_imagenet100_dual_knn_prc=2.5"' \
    ++method_kwargs.loss.type="dual_knn" \
    ++method_kwargs.prediction_precision=2.5 \
    ++seed=$1 

python3 main_pretrain.py \
    --config-path scripts/pretrain/imagenet-100/ \
    --config-name sphereprob.yaml \
    ++name='"sphereprob_imagenet100_single_sample"' \
    ++method_kwargs.loss.type="single_sample" \
    ++seed=$1 

python3 main_pretrain.py \
    --config-path scripts/pretrain/imagenet-100/ \
    --config-name sphereprob.yaml \
    ++name='"sphereprob_imagenet100_single_gauss"' \
    ++method_kwargs.loss.type="single_gauss" \
    ++seed=$1 

python3 main_pretrain.py \
    --config-path scripts/pretrain/cifar/ \
    --config-name sphereprob.yaml \
    ++name='"sphereprob_imagenet100_single_knn"' \
    ++method_kwargs.loss.type="single_knn" \
    ++seed=$1 
