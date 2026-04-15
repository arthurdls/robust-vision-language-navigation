# OpenVLA `--device` flag

**Date:** 2026-04-15
**Status:** Implemented

## Goal

Let `scripts/start_server.py` run OpenVLA without a dedicated GPU, or with the weights split between GPU VRAM and system RAM, without quantization.

## Interface

Single flag on `scripts/start_server.py`:

```
--device {cuda, cpu, auto}   (default: cuda)
```

- `cuda`: existing behavior. bf16 weights, `attn_implementation="flash_attention_2"`, pinned to `--gpu-id`.
- `cpu`: fp32 weights on CPU. No flash-attn. Slow (tens of seconds per action for a 7B model) but needs no GPU.
- `auto`: `device_map="auto"` via accelerate. bf16 weights, layers split across GPU and CPU RAM by available VRAM. No flash-attn (incompatible with offload). Requires a CUDA GPU.

`--gpu-id` stays for backward compat and is only meaningful with `--device cuda`. Supplying it with `cpu` or `auto` logs a warning and is ignored.

## Implementation

Changes are confined to two files plus the server env:

- `scripts/start_server.py`: add the `--device` arg, pass it through `cfg["device"]`, warn on stray `--gpu-id`.
- `src/rvln/server/openvla.py` (`OpenVLAActionAgent.__init__`): branch on `cfg["device"]` when building the `from_pretrained` kwargs and deciding whether to call `.to(device)`. Store `self.compute_dtype` so `act()` can cast inputs correctly. For `auto`, infer `self.device` from `next(self.model.parameters()).device` so inputs go to the embedding layer's device.
- `act()`: replace the hardcoded `torch.bfloat16` cast with `self.compute_dtype`.
- `rvln-server_env.yml`: add `accelerate` (required for `device_map="auto"`).

## Error handling

- `cuda` or `auto` selected but `torch.cuda.is_available()` is False: raise `RuntimeError` at startup with a message pointing at `--device cpu`.
- Unknown `--device` value: argparse rejects it at parse time.
- `--gpu-id` with non-cuda device: log warning, continue.

## Non-goals

- Quantization (user explicitly ruled out).
- CPU flag on any other script. Other scripts call the server over HTTP and don't load weights.
- Reworking the env file to support a CPU-only install (flash-attn is still a hard dep for `cuda` mode; a pure-CPU user can pip-install the minimal subset or remove `flash-attn` from the env manually).

## Testing

No existing test infra for the server. Manual verification:

```
# cuda (baseline)
python scripts/start_server.py --device cuda

# cpu
python scripts/start_server.py --device cpu

# auto
python scripts/start_server.py --device auto
```

For each, POST a sample payload to `/predict` and confirm a 200 with a numeric action.
