# Monodepth encoder using dynamic filtering
## Setup environment
```bash
pip install timm pyyaml ipdb
```

```bash
git clone https://github.com/NVIDIA/apex
cd apex
pip install -v --disable-pip-version-check --no-cache-dir --global-option="--cpp_ext" --global-option="--cuda_ext" ./
```

```bash
cd ddf
python setup.py install
mv build/lib*/* .
```
## Train

```bash
python train.py  --split oxford_all \
                  --data_path /workspace/datasets/oxfordrobocar \
                  --log_dir /workspace/datasets/log/DFDepth \
                  --model_name dfdepth_v2 \
                  --batch_size 4 \
                  --num_epochs 20 \
                  --use_day_pose \
                  --only_day_reprojection \
                  --dida_level 4 \
                  --train_day_only \
                  --encoder resnet_ddf
```



