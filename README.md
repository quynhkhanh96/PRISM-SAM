# prism-sam (TMLR revision codebase)

```
third_party/segment_anything/   modified SAM1 (4 masks, attention cache) — single copy
prismsam/
  lora.py        LoRALinear (legacy-compatible names .base/.A/.B), LoRAFusedQKV (ViT qkv)
  adapt.py       apply_adaptation(model, cfg): tok | twt | nontwt | dec | enc  x  lora | full
  variants.py    canonical R1/R5 configs, expected #params, experiment-name fragment
  models.py      build_model(cfg), save_trainable / load_finetuned (new + legacy ckpts)
  losses.py      SAMMultimaskLoss (unchanged)
tests/
  test_variants.py   pre-flight: counts, identity-at-init, gradient flow, ckpt I/O
```

Run before any training:

```
python tests/test_variants.py --device cuda                         # full, ViT-L
python tests/test_variants.py --model_type vit_b --skip counts      # CPU, small RAM
```

## R1/R5 variants (SAM1 ViT-L, mask tokens included)

| name | trainable | config |
|---|---|---|
| tok | 1,024 | mask tokens |
| twt-lora (PRISM-SAM) | 95,232 | TWT q/k/v/o, scope=full, r8a8 |
| twt-full | 3,292,288 | all TWT |
| nontwt-full | 634,208 | hypernet MLPs + output upscaling (IoU head, iou_token frozen) |
| nontwt-lora | 95,488 | LoRA on hypernet Linear layers, r18a18; upscaling frozen |
| enc-lora | 99,328 | ViT last 2 blocks, independent q/k/v/o adapters, r6a6 |
| dec-full (existing) | 4,058,340 | whole decoder; 132,612 (IoU head) never receive gradient |

Training code must give the optimizer `adapt.trainable_parameters(model)`
(all trainable tensors), not `mask_decoder.parameters()`.
