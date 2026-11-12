"""Reconstruct an evaluated MiniLM checkpoint from the pinned base and its patch."""

import argparse
import hashlib
import json
from pathlib import Path
import shutil
import struct
import tempfile

ROOT = Path(__file__).resolve().parents[1]


def digest(data):
    return hashlib.sha256(data).hexdigest()


def parse_header(data):
    size = struct.unpack("<Q", data[:8])[0]
    end = 8 + size
    return end, {
        k: v for k, v in json.loads(data[8:end]).items() if k != "__metadata__"
    }


def main():
    patches = ROOT / "patches"
    manifest = json.loads((patches / "manifest.json").read_text())
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("checkpoint", choices=sorted(manifest["checkpoints"]))
    parser.add_argument(
        "--base", type=Path, help="Local model.safetensors or its parent directory"
    )
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()

    if args.base is None:
        from huggingface_hub import hf_hub_download

        args.base = Path(
            hf_hub_download(
                manifest["base_model"],
                "model.safetensors",
                revision=manifest["base_revision"],
            )
        )
    base_path = args.base / "model.safetensors" if args.base.is_dir() else args.base
    base = base_path.read_bytes()
    source_hash = digest(base)
    if source_hash not in {manifest["base_sha256"], manifest["public_base_sha256"]}:
        raise ValueError(
            "Base weights do not match the pinned public or evaluated file"
        )
    public_base = source_hash == manifest["public_base_sha256"]
    base_start, base_tensors = parse_header(base)
    record = manifest["checkpoints"][args.checkpoint]
    directory = patches / args.checkpoint
    header = (directory / "header.bin").read_bytes()
    payload = (directory / "replacement.bin").read_bytes()
    if (digest(header), digest(payload)) != (
        record["header_sha256"],
        record["payload_sha256"],
    ):
        raise ValueError("Patch checksum mismatch")
    start, tensors = parse_header(header)
    expected_extra = {"embeddings.position_ids"} if public_base else set()
    if (
        start != len(header)
        or set(base_tensors) - set(tensors) != expected_extra
        or not set(tensors) <= set(base_tensors)
    ):
        raise ValueError("Unexpected checkpoint header")

    if args.output.exists():
        raise FileExistsError(f"Choose a new output directory: {args.output}")
    args.output.parent.mkdir(parents=True, exist_ok=True)
    with tempfile.TemporaryDirectory(dir=args.output.parent) as tmp:
        staged = Path(tmp) / "model"
        shutil.copytree(directory / "config", staged)
        checkpoint = staged / "model.safetensors"
        with checkpoint.open("wb") as stream:
            stream.write(header)
            for name, info in sorted(
                tensors.items(), key=lambda item: item[1]["data_offsets"][0]
            ):
                before = base_tensors[name]
                if name in record["tensors"]:
                    lo, hi = record["tensors"][name]
                    value = payload[lo:hi]
                else:
                    lo, hi = before["data_offsets"]
                    value = base[base_start + lo : base_start + hi]
                    if (
                        public_base
                        and before["dtype"] == "F32"
                        and info["dtype"] == "F16"
                    ):
                        import numpy as np

                        value = (
                            np.frombuffer(value, dtype="<f4").astype("<f2").tobytes()
                        )
                lo, hi = info["data_offsets"]
                if len(value) != hi - lo or info["shape"] != before["shape"]:
                    raise ValueError(
                        f"Unexpected tensor shape or byte length for {name}"
                    )
                stream.write(value)
        actual = digest(checkpoint.read_bytes())
        if actual != record["sha256"]:
            raise ValueError("Reconstructed checkpoint checksum mismatch")
        staged.rename(args.output)
    print(f"{args.checkpoint} restored and verified: {actual}")


if __name__ == "__main__":
    main()
