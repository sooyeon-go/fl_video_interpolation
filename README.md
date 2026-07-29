# First/Last Frame Video Generation

One CLI for:

- OmniWeaving (`interpolation`)
- LTX-2.3 (`KeyframeInterpolationPipeline`)
- Wan 2.2 I2V-A14B (`WanImageToVideoPipeline.last_image`)

## Model paths

Default download root:

```text
/data/shared-vilab/pretrained_models
```

Expected layout after `./prepare_models.sh all`:

```text
/data/shared-vilab/pretrained_models/
  Wan2.2-I2V-A14B-Diffusers/   # Diffusers format for FLF
  HY-OmniWeaving/              # OmniWeaving + encoders
  LTX-2.3/
    gemma-3-12b/               # LTX text encoder
    ... existing LTX weights ...
```

`Wan2.2-I2V-A14B` (original format with `high_noise_model/`) is kept as-is.
FLF for Wan uses the separate Diffusers directory.

## Conda environments

```bash
chmod +x ./*.sh

# Create all conda envs
./setup_envs.sh all

# Hugging Face login (needed for gated Omni encoder)
conda run -n flf-tools hf auth login

# Download models into /data/shared-vilab/pretrained_models
./prepare_models.sh all
```

Or everything at once:

```bash
./setup_all.sh
```

Created envs:

| Env | Purpose |
|---|---|
| `flf-tools` | `hf` / ModelScope download CLIs |
| `flf-wan` | Wan 2.2 Diffusers FLF |
| `flf-omni` | OmniWeaving interpolation |
| `flf-ltx` | LTX-2.3 keyframe interpolation |

Activate manually if needed:

```bash
conda activate flf-wan
conda activate flf-omni
conda activate flf-ltx
```

## Run

`run_flf.sh` automatically picks the matching conda env.

```bash
./run_flf.sh ltx first.png last.png outputs/ltx.mp4 \
  "A person naturally turns around and walks toward the door." \
  --frames 81 --width 832 --height 512

./run_flf.sh wan first.png last.png outputs/wan.mp4 \
  "A person naturally turns around and walks toward the door." \
  --frames 81 --width 832 --height 480

./run_flf.sh omni first.png last.png outputs/omni.mp4 \
  "A person naturally turns around and walks toward the door." \
  --frames 81 --num-gpus 8 --think
```

Run `python generate_flf.py --help` for all options.
