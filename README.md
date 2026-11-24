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
cp local_paths.example.yaml local_paths.yaml
pip install pyyaml pandas pyarrow
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

## R2 — attention enrichment across adaptation sites

```
python scripts/run_r2.py --dry_run
python scripts/run_r2.py --gpus 0,1,2 --sanity     # tok0 identity check + attention readouts
python scripts/r2_table.py                         # -> <results_root>/analysis/R2/r2_table.md
```

- `prismsam/attention.py`: Section 1 = verbatim A_enrich_remote code from the
  pre-revision `attention_alignment_analysis.py`; Section 2 = raw per-head
  attention mass on remote / clicked regions for every token->image layer.
- `scripts/eval_attn.py`: per (instance, prompt) attention file, paired with
  the PR files on `instance_id` / `prompt_id`.
- `scripts/r2_sanity_tok0.py`: tok0 must reproduce SAM1's TWT bit for bit.
- `prismsam/r2.py`: experiment list + old S4.2 numbers (regression check).
- `prismsam/bootstrap.py`: paired hierarchical bootstrap (seeds, then instances).