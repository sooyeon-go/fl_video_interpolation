#!/usr/bin/env python3
"""Generate a video conditioned on first and last frames."""

from __future__ import annotations

import argparse
import subprocess
import sys
from pathlib import Path


DEFAULT_LTX_MODEL = Path("/data/shared-vilab/pretrained_models/LTX-2.3")
DEFAULT_WAN_MODEL = Path(
    "/data/shared-vilab/pretrained_models/Wan2.2-I2V-A14B-Diffusers"
)
DEFAULT_OMNI_MODEL = Path(
    "/data/shared-vilab/pretrained_models/HY-OmniWeaving"
)
DEFAULT_OMNI_REPO = Path(__file__).resolve().parent / "third_party" / "OmniWeaving"

DEFAULT_NEGATIVE_PROMPT = (
    "low quality, blurry, distorted, deformed anatomy, flicker, jitter, "
    "duplicate objects, subtitles, watermark"
)


def existing_file(value: str) -> Path:
    path = Path(value).expanduser().resolve()
    if not path.is_file():
        raise argparse.ArgumentTypeError(f"File does not exist: {path}")
    return path


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="First/last-frame video generation with OmniWeaving, LTX-2.3, or Wan 2.2."
    )
    parser.add_argument("--model", choices=("omni", "ltx", "wan"), required=True)
    parser.add_argument("--first-frame", type=existing_file, required=True)
    parser.add_argument("--last-frame", type=existing_file, required=True)
    parser.add_argument("--prompt", required=True)
    parser.add_argument("--negative-prompt", default=DEFAULT_NEGATIVE_PROMPT)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--frames", type=int, default=81)
    parser.add_argument("--fps", type=float, default=16.0)
    parser.add_argument("--steps", type=int, default=None)
    parser.add_argument("--height", type=int, default=480)
    parser.add_argument("--width", type=int, default=832)
    parser.add_argument("--model-path", type=Path, default=None)
    parser.add_argument("--offload", action=argparse.BooleanOptionalAction, default=True)

    parser.add_argument("--repo-path", type=Path, default=None)
    parser.add_argument("--python", default=sys.executable)
    parser.add_argument("--num-gpus", type=int, default=1)

    parser.add_argument("--ltx-gemma-path", type=Path)
    parser.add_argument("--ltx-distilled-lora", type=Path)
    parser.add_argument("--ltx-upscaler", type=Path)
    parser.add_argument("--ltx-keyframe-strength", type=float, default=1.0)

    parser.add_argument("--wan-guidance-scale", type=float, default=3.5)
    parser.add_argument(
        "--think",
        action="store_true",
        help="Enable OmniWeaving reasoning-based prompt enhancement.",
    )
    return parser


def require_path(path: Path, description: str) -> Path:
    path = path.expanduser().resolve()
    if not path.exists():
        raise FileNotFoundError(f"{description} does not exist: {path}")
    return path


def find_one(root: Path, patterns: tuple[str, ...], description: str) -> Path:
    for pattern in patterns:
        matches = sorted(root.glob(pattern))
        if matches:
            return matches[0]
    patterns_text = ", ".join(patterns)
    raise FileNotFoundError(
        f"Could not find {description} under {root}. Tried: {patterns_text}"
    )


def run_command(command: list[str], cwd: Path | None = None) -> None:
    printable = " ".join(map(str, command))
    print(f"Running: {printable}", flush=True)
    subprocess.run(command, cwd=cwd, check=True)


def run_omni(args: argparse.Namespace) -> None:
    model_path = require_path(
        args.model_path or DEFAULT_OMNI_MODEL, "OmniWeaving model"
    )
    repo_path = require_path(
        args.repo_path or DEFAULT_OMNI_REPO, "OmniWeaving repository"
    )
    generate_script = require_path(repo_path / "generate.py", "OmniWeaving generate.py")

    command = [
        args.python,
        str(generate_script),
        "--task",
        "interpolation",
        "--model_path",
        str(model_path),
        "--ref_image_paths",
        str(args.first_frame),
        str(args.last_frame),
        "--prompt",
        args.prompt,
        "--negative_prompt",
        args.negative_prompt,
        "--video_length",
        str(args.frames),
        "--fps",
        str(round(args.fps)),
        "--seed",
        str(args.seed),
        "--output_path",
        str(args.output),
        "--offloading",
        str(args.offload).lower(),
    ]
    if args.steps is not None:
        command.extend(["--num_inference_steps", str(args.steps)])
    if args.think:
        command.append("--think")
    if args.num_gpus > 1:
        command = [
            args.python,
            "-m",
            "torch.distributed.run",
            f"--nproc_per_node={args.num_gpus}",
            str(generate_script),
            *command[2:],
        ]
    run_command(command, cwd=repo_path)


def run_ltx(args: argparse.Namespace) -> None:
    if args.frames % 8 != 1:
        raise ValueError("LTX-2.3 requires --frames to equal 8k+1 (for example, 81 or 121).")
    if args.height % 64 or args.width % 64:
        raise ValueError("The two-stage LTX pipeline requires width and height divisible by 64.")

    model_path = require_path(args.model_path or DEFAULT_LTX_MODEL, "LTX-2.3 model")
    checkpoint = find_one(
        model_path,
        ("ltx-2.3-22b-dev.safetensors", "*22b-dev*.safetensors"),
        "LTX development checkpoint",
    )
    distilled_lora = (
        require_path(args.ltx_distilled_lora, "LTX distilled LoRA")
        if args.ltx_distilled_lora
        else find_one(
            model_path,
            ("*distilled-lora-384-1.1*.safetensors", "*distilled-lora*.safetensors"),
            "LTX distilled LoRA",
        )
    )
    upscaler = (
        require_path(args.ltx_upscaler, "LTX spatial upscaler")
        if args.ltx_upscaler
        else find_one(
            model_path,
            ("*spatial-upscaler-x2-1.1*.safetensors", "*spatial-upscaler-x2*.safetensors"),
            "LTX spatial upscaler",
        )
    )
    if args.ltx_gemma_path:
        gemma_path = require_path(args.ltx_gemma_path, "Gemma text encoder")
    else:
        gemma_path = find_one(
            model_path,
            ("gemma-3-12b*", "text_encoder/gemma-3-12b*", "gemma*"),
            "Gemma text encoder directory; pass --ltx-gemma-path explicitly",
        )

    command = [
        args.python,
        "-m",
        "ltx_pipelines.keyframe_interpolation",
        "--checkpoint-path",
        str(checkpoint),
        "--distilled-lora",
        str(distilled_lora),
        "1.0",
        "--spatial-upsampler-path",
        str(upscaler),
        "--gemma-root",
        str(gemma_path),
        "--prompt",
        args.prompt,
        "--negative-prompt",
        args.negative_prompt,
        "--image",
        str(args.first_frame),
        "0",
        str(args.ltx_keyframe_strength),
        "--image",
        str(args.last_frame),
        str(args.frames - 1),
        str(args.ltx_keyframe_strength),
        "--height",
        str(args.height),
        "--width",
        str(args.width),
        "--num-frames",
        str(args.frames),
        "--frame-rate",
        str(args.fps),
        "--seed",
        str(args.seed),
        "--output-path",
        str(args.output),
    ]
    if args.steps is not None:
        command.extend(["--num-inference-steps", str(args.steps)])
    if args.offload:
        command.extend(["--offload", "cpu"])
    run_command(command)


def resize_and_crop(image, width: int, height: int):
    from PIL import Image, ImageOps

    return ImageOps.fit(
        image.convert("RGB"),
        (width, height),
        method=Image.Resampling.LANCZOS,
    )


def run_wan(args: argparse.Namespace) -> None:
    if args.frames % 4 != 1:
        raise ValueError("Wan 2.2 requires --frames to equal 4k+1 (for example, 81).")
    if args.height % 16 or args.width % 16:
        raise ValueError("Wan 2.2 width and height must be divisible by 16.")

    import torch
    from diffusers import WanImageToVideoPipeline
    from diffusers.utils import export_to_video
    from PIL import Image

    model_source = str(args.model_path or DEFAULT_WAN_MODEL)
    local_model = Path(model_source).expanduser()
    if local_model.exists() and not (local_model / "model_index.json").is_file():
        raise ValueError(
            f"{local_model} is the original Wan format, not Diffusers format. "
            "Use the converted Wan2.2-I2V-A14B-Diffusers directory or pass "
            "--model-path Wan-AI/Wan2.2-I2V-A14B-Diffusers."
        )

    first_frame = resize_and_crop(
        Image.open(args.first_frame), args.width, args.height
    )
    last_frame = resize_and_crop(
        Image.open(args.last_frame), args.width, args.height
    )

    pipeline = WanImageToVideoPipeline.from_pretrained(
        model_source,
        torch_dtype=torch.bfloat16,
    )
    if args.offload:
        pipeline.enable_model_cpu_offload()
    else:
        pipeline.to("cuda")

    generator = torch.Generator(device="cuda").manual_seed(args.seed)
    result = pipeline(
        image=first_frame,
        last_image=last_frame,
        prompt=args.prompt,
        negative_prompt=args.negative_prompt,
        height=args.height,
        width=args.width,
        num_frames=args.frames,
        num_inference_steps=args.steps or 40,
        guidance_scale=args.wan_guidance_scale,
        generator=generator,
    ).frames[0]
    export_to_video(result, str(args.output), fps=args.fps)


def main() -> None:
    args = build_parser().parse_args()
    args.output = args.output.expanduser().resolve()
    args.output.parent.mkdir(parents=True, exist_ok=True)

    runners = {"omni": run_omni, "ltx": run_ltx, "wan": run_wan}
    runners[args.model](args)
    print(f"Saved video: {args.output}")


if __name__ == "__main__":
    main()
