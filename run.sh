####################### SSL pre-train ResNet-50
LOG_PATH="./logs/backbone"
CUDA_VISIBLE_DEVICES=0,1 TORCH_DISTRIBUTED_DEBUG=DETAIL python tools/backbone_train.py \
    -a resnet50 --arch-version "v32" --data-name "SevenPCBaseDataset" --data-path "./data/7PC" \
    --mean 0.7833 0.6712 0.6026 --std 0.2139 0.2472 0.2571 \
    --epochs 400 -b 96 -lr 1e-6 -j 4 \
    --img-sz 224 224 \
    --num-labels 8 \
    --proj-dim 128 --temperature 0.1 \
    --log-path "${LOG_PATH}" \
    --proj-name "sm3_r50_backbone" --arch-weights "IMAGENET1K_V1" \
    --amp

EPOCHS=(49 99 149 199 249 299 349 399)
for epoch in ${EPOCHS[@]}
do
    CUDA_VISIBLE_DEVICES=0 TORCH_DISTRIBUTED_DEBUG=DETAIL python tools/backbone_eval.py \
        -a resnet50 --data-name "SevenPCBaseDataset" --data-path "./data/7PC" \
        --mean 0.7833 0.6712 0.6026 --std 0.2139 0.2472 0.2571 \
        --epochs 50 -b 128 -lr 1e-3 -j 4 \
        --img-sz 224 224 \
        --num-labels 8 \
        --pretrain-path "${LOG_PATH}/ckp_${epoch}.pth" \
        --finetune "fc" \
        --log-path "${LOG_PATH}/test_${epoch}" \
        --proj-name "sm3_r50_backbone_eval" --amp
done


####################### MLC pre-train ResNet-50
LOG_PATH="./logs/mlc_train"
CUDA_VISIBLE_DEVICES=0 TORCH_DISTRIBUTED_DEBUG=DETAIL python tools/mlc_train.py \
    -a resnet50 --data-name "SevenPCBaseDataset" --data-path "./data/7PC" \
    --mean 0.7833 0.6712 0.6026 --std 0.2139 0.2472 0.2571 \
    --epochs 150 -b 256 -lr 1e-4 -j 4 \
    --img-sz 224 224 \
    --num-labels 8 \
    --temperature 1 \
    --mlc-proj "v4" --mlc-proj-dim 512 \
    --num-heads 1 --sa-dim-ff 128 --sa-dropout 0.1 \
    --extractor-proj-dim 128 --extractor-weights "./logs/backbone/ckp_399.pth" \
    --log-path "${LOG_PATH}" \
    --proj-name "SM3_MLC_train_v4_r50"

EPOCHS=(49 99 149)
for epoch in ${EPOCHS[@]}
do
    CUDA_VISIBLE_DEVICES=0 TORCH_DISTRIBUTED_DEBUG=DETAIL python tools/mlc_eval.py \
        -a resnet50 --data-name "SevenPCBaseDataset" --data-path "./data/7PC" \
        --mean 0.7833 0.6712 0.6026 --std 0.2139 0.2472 0.2571 \
        --epochs 100 -b 128 -lr 1e-3 -j 4 \
        --img-sz 224 224 \
        --num-labels 8 \
        --mlc-proj "v4" --mlc-proj-dim 512 \
        --num-heads 1 --sa-dim-ff 128 --sa-dropout 0.1 \
        --extractor-proj-dim 128 \
        --pretrain-path "${LOG_PATH}/ckp_${epoch}.pth" \
        --finetune "projector" \
        --log-path "${LOG_PATH}/test_${epoch}" \
        --proj-name "SM3_MLC_eval_v4_r50"
done