# Kikuchi SR GAN PyTorch

PyTorch version for one-to-one Kikuchi super-resolution training.

This folder keeps the training functions from the original EBSD GAN workflow in a PyTorch form:

- raw data loader
- generator and discriminator
- adversarial training loop
- CSV training log
- checkpoints
- fixed-sample visualization during training
- TensorBoard scalars/images
- tqdm epoch progress
- automatic `loss_curves.png`

The paired data are read directly from:

```text
C:\Users\WHJ\Desktop\kikuchi-super resolution
```

Default pairs:

| Area | LowR input | HighR target | Pairs |
| --- | --- | --- | ---: |
| Area 1 | `20260512_Cu_Area 1_OIM Map 2.up2` `(117, 117)` | `20260512_Cu_Area 1_OIM Map 1.up2` `(470, 470)` | 42510 |
| Area 2 | `20260512_Cu_Area 2_OIM Map 2.up2` `(117, 117)` | `20260512_Cu_Area 2_OIM Map 1.up2` `(470, 470)` | 28302 |

Total full paired dataset: 70812 samples.

No CLAHE, histogram equalization, filtering, cropping, or pre-resize is applied to the raw patterns. The only data conversion is `uint16 -> float32 / 65535` for neural-network training.

By default, training generates circular detector masks for the central valid Kikuchi area:

- LowR mask: `(117, 117)`
- HighR mask: `(470, 470)`
- default radius: `0.49 * min(height, width)`

The masked background is zeroed for the network input/target, and the pixel loss plus discriminator input are computed on the valid circular region so the outer background does not dominate the GAN.

## Run

Use the existing PyTorch conda environment:

```powershell
cd D:\project\EBSD2026\Kikuchi_SR_GAN_PyTorch
D:\anaconda3\envs\torch\python.exe train_srgan.py --epochs 500 --batch-size 4 --amp
```

Useful outputs:

```text
runs/kikuchi_srgan_masked_full_500_tb/manifest.json
runs/kikuchi_srgan_masked_full_500_tb/training_log.csv
runs/kikuchi_srgan_masked_full_500_tb/loss_curves.png
runs/kikuchi_srgan_masked_full_500_tb/tensorboard/
runs/kikuchi_srgan_masked_full_500_tb/previews/raw_pairs.png
runs/kikuchi_srgan_masked_full_500_tb/masks/low_mask.png
runs/kikuchi_srgan_masked_full_500_tb/masks/high_mask.png
runs/kikuchi_srgan_masked_full_500_tb/previews/epoch_000.png
runs/kikuchi_srgan_masked_full_500_tb/checkpoints/latest.pt
```

Open TensorBoard:

```powershell
D:\anaconda3\envs\torch\Scripts\tensorboard.exe --logdir D:\project\EBSD2026\Kikuchi_SR_GAN_PyTorch\runs\kikuchi_srgan_masked_full_500_tb\tensorboard --port 6006
```

For a quick pipeline test:

```powershell
D:\anaconda3\envs\torch\python.exe train_srgan.py --epochs 1 --batch-size 2 --max-steps-per-epoch 2 --out-dir runs/smoke --amp
```
