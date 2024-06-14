import argparse
import os
import re

import torch

from pipeline.pixart_controlnet_transformer import PixArtControlNetAdapterModel

def main(args):
    all_state_dict = torch.load(args.orig_ckpt_path, map_location='cpu')
    state_dict = all_state_dict.pop("state_dict")

    converted_state_dict = {}

    controlnet_layers_found = 0
    patternForControlnetKeys = re.compile(r"^controlnet\.(\d+)\.")
    for key in state_dict.keys():
        match = patternForControlnetKeys.match(key)
        if not match:
            continue
        
        depth = int(match.group(1))
        
        controlnet_layers_found = max(controlnet_layers_found, depth + 1)

    print(f"Controlnet layers found: {controlnet_layers_found}")

    # map new dict
    for depth in range(controlnet_layers_found):
        # Transformer blocks.
        converted_state_dict[f"controlnet_blocks.{depth}.transformer_block.scale_shift_table"] = state_dict.pop(
            f"controlnet.{depth}.copied_block.scale_shift_table"
        )

        # Attention is all you need 🤘

        # Self attention.
        q, k, v = torch.chunk(state_dict.pop(f"controlnet.{depth}.copied_block.attn.qkv.weight"), 3, dim=0)
        q_bias, k_bias, v_bias = torch.chunk(state_dict.pop(f"controlnet.{depth}.copied_block.attn.qkv.bias"), 3, dim=0)
        converted_state_dict[f"controlnet_blocks.{depth}.transformer_block.attn1.to_q.weight"] = q
        converted_state_dict[f"controlnet_blocks.{depth}.transformer_block.attn1.to_q.bias"] = q_bias
        converted_state_dict[f"controlnet_blocks.{depth}.transformer_block.attn1.to_k.weight"] = k
        converted_state_dict[f"controlnet_blocks.{depth}.transformer_block.attn1.to_k.bias"] = k_bias
        converted_state_dict[f"controlnet_blocks.{depth}.transformer_block.attn1.to_v.weight"] = v
        converted_state_dict[f"controlnet_blocks.{depth}.transformer_block.attn1.to_v.bias"] = v_bias
        # Projection.
        converted_state_dict[f"controlnet_blocks.{depth}.transformer_block.attn1.to_out.0.weight"] = state_dict.pop(
            f"controlnet.{depth}.copied_block.attn.proj.weight"
        )
        converted_state_dict[f"controlnet_blocks.{depth}.transformer_block.attn1.to_out.0.bias"] = state_dict.pop(
            f"controlnet.{depth}.copied_block.attn.proj.bias"
        )

        # Feed-forward.
        converted_state_dict[f"controlnet_blocks.{depth}.transformer_block.ff.net.0.proj.weight"] = state_dict.pop(
            f"controlnet.{depth}.copied_block.mlp.fc1.weight"
        )
        converted_state_dict[f"controlnet_blocks.{depth}.transformer_block.ff.net.0.proj.bias"] = state_dict.pop(
            f"controlnet.{depth}.copied_block.mlp.fc1.bias"
        )
        converted_state_dict[f"controlnet_blocks.{depth}.transformer_block.ff.net.2.weight"] = state_dict.pop(
            f"controlnet.{depth}.copied_block.mlp.fc2.weight"
        )
        converted_state_dict[f"controlnet_blocks.{depth}.transformer_block.ff.net.2.bias"] = state_dict.pop(
            f"controlnet.{depth}.copied_block.mlp.fc2.bias"
        )

        # Cross-attention.
        q = state_dict.pop(f"controlnet.{depth}.copied_block.cross_attn.q_linear.weight")
        q_bias = state_dict.pop(f"controlnet.{depth}.copied_block.cross_attn.q_linear.bias")
        k, v = torch.chunk(state_dict.pop(f"controlnet.{depth}.copied_block.cross_attn.kv_linear.weight"), 2, dim=0)
        k_bias, v_bias = torch.chunk(state_dict.pop(f"controlnet.{depth}.copied_block.cross_attn.kv_linear.bias"), 2, dim=0)

        converted_state_dict[f"controlnet_blocks.{depth}.transformer_block.attn2.to_q.weight"] = q
        converted_state_dict[f"controlnet_blocks.{depth}.transformer_block.attn2.to_q.bias"] = q_bias
        converted_state_dict[f"controlnet_blocks.{depth}.transformer_block.attn2.to_k.weight"] = k
        converted_state_dict[f"controlnet_blocks.{depth}.transformer_block.attn2.to_k.bias"] = k_bias
        converted_state_dict[f"controlnet_blocks.{depth}.transformer_block.attn2.to_v.weight"] = v
        converted_state_dict[f"controlnet_blocks.{depth}.transformer_block.attn2.to_v.bias"] = v_bias

        converted_state_dict[f"controlnet_blocks.{depth}.transformer_block.attn2.to_out.0.weight"] = state_dict.pop(
            f"controlnet.{depth}.copied_block.cross_attn.proj.weight"
        )
        converted_state_dict[f"controlnet_blocks.{depth}.transformer_block.attn2.to_out.0.bias"] = state_dict.pop(
            f"controlnet.{depth}.copied_block.cross_attn.proj.bias"
        )

        # The before proj layer
        if depth == 0:
            print(f"\tAdding before_proj layer for depth 0")
            converted_state_dict[f"controlnet_blocks.{depth}.before_proj.weight"] = state_dict.pop("controlnet.0.before_proj.weight")
            converted_state_dict[f"controlnet_blocks.{depth}.before_proj.bias"] = state_dict.pop("controlnet.0.before_proj.bias")

        # The after proj layer
        converted_state_dict[f"controlnet_blocks.{depth}.after_proj.weight"] = state_dict.pop(f"controlnet.{depth}.after_proj.weight")
        converted_state_dict[f"controlnet_blocks.{depth}.after_proj.bias"] = state_dict.pop(f"controlnet.{depth}.after_proj.bias")

    controlnet = PixArtControlNetAdapterModel()
    controlnet.load_state_dict(converted_state_dict, strict=True)

    num_model_params = sum(p.numel() for p in controlnet.parameters())
    print(f"Total number of controlnet parameters: {num_model_params}")
    
    controlnet.save_pretrained(os.path.join(args.dump_path, "controlnet"))

if __name__ == "__main__":
    parser = argparse.ArgumentParser()

    parser.add_argument("--orig_ckpt_path", default=None, type=str, required=False, help="Path to the checkpoint to convert.")
    parser.add_argument("--dump_path", default=None, type=str, required=True, help="Path to the output pipeline.")
        
    args = parser.parse_args()
    main(args)
