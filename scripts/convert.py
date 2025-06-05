import os
import json
import shutil
import argparse

import torch
from safetensors.torch import load_file, save_file


def convert_safetensor(
    input_path: str, output_path: str, target_dtype: torch.dtype
) -> None:
    print(f"Converting: {input_path} -> {output_path} (dtype={target_dtype})")
    tensor_dict = load_file(input_path)

    converted_dict = {}
    for name, tensor in tensor_dict.items():
        print(f"Converting tensor: {name} to {target_dtype}")
        converted_tensor = tensor.to(target_dtype)
        converted_dict[name] = converted_tensor

    print(f"Saving: {input_path} -> {output_path}")
    save_file(converted_dict, output_path)


def process_model_folder(
    src_folder: str,
    dst_folder: str,
    target_dtype: torch.dtype,
) -> None:
    if not os.path.isdir(src_folder):
        raise FileNotFoundError("Source folder does not exists.")

    os.makedirs(dst_folder, exist_ok=True)

    index_filename = "model.safetensors.index.json"
    index_src_path = os.path.join(src_folder, index_filename)
    if not os.path.isfile(index_src_path):
        raise FileNotFoundError(f"{index_src_path} does not exists.")

    with open(index_src_path, "r", encoding="utf-8") as f:
        index_data = json.load(f)

    safetensor_list = set()
    weight_map = index_data.get("weight_map", {})
    for _, shard_path in weight_map.items():
        safetensor_list.add(shard_path)

    print(safetensor_list)

    for weights_file in safetensor_list:
        abs_src_path = os.path.join(src_folder, weights_file)
        abs_dst_path = os.path.join(dst_folder, weights_file)
        convert_safetensor(abs_src_path, abs_dst_path, target_dtype)

    # walk in src fold and copy extension with txt and json
    for root, _, files in os.walk(src_folder):
        for filename in files:
            if not filename.endswith((".txt", ".json")):
                continue
            print(f"Copying: {filename}")
            abs_src_path = os.path.join(root, filename)
            rel_path = os.path.relpath(abs_src_path, src_folder)
            abs_dst_path = os.path.join(dst_folder, rel_path)
            shutil.copy2(abs_src_path, abs_dst_path)


def parse_dtype(dtype_str: str) -> torch.dtype:
    dtype_str = dtype_str.lower()
    if dtype_str == "fp32":
        return torch.float32
    elif dtype_str == "bf16":
        return torch.bfloat16
    elif dtype_str == "fp16":
        return torch.float16
    else:
        raise ValueError("Unknown dtype")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--src", "-s", required=True, help="source folder")
    parser.add_argument("--dst", "-d", required=True, help="destination folder")
    parser.add_argument(
        "--dtype", "-t", required=True, choices=["fp32", "bf16", "fp16"]
    )
    args = parser.parse_args()

    src_folder = args.src
    dst_folder = args.dst
    target_dtype = parse_dtype(args.dtype)

    print(f"Source folder: {src_folder}")
    print(f"Destination folder: {dst_folder}")
    print(f"Target dtype: {target_dtype}")

    process_model_folder(src_folder, dst_folder, target_dtype)
    print("Processing complete.")
