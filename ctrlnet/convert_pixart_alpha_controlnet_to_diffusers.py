import argparse
import os

import torch

from diffusers import Transformer2DModel

def main(args):
    all_state_dict = torch.load(args.orig_ckpt_path, map_location='cpu')
    state_dict = all_state_dict.pop("state_dict")

    converted_state_dict = {}

    for depth in range(28):
        # Transformer blocks.
        converted_state_dict[f"controlnet_transformer_blocks.{depth}.scale_shift_table"] = state_dict.pop(
            f"controlnet.{depth}.copied_block.scale_shift_table"
        )

        # Attention is all you need 🤘

        # Self attention.
        q, k, v = torch.chunk(state_dict.pop(f"controlnet.{depth}.copied_block.attn.qkv.weight"), 3, dim=0)
        q_bias, k_bias, v_bias = torch.chunk(state_dict.pop(f"controlnet.{depth}.copied_block.attn.qkv.bias"), 3, dim=0)
        converted_state_dict[f"controlnet_transformer_blocks.{depth}.attn1.to_q.weight"] = q
        converted_state_dict[f"controlnet_transformer_blocks.{depth}.attn1.to_q.bias"] = q_bias
        converted_state_dict[f"controlnet_transformer_blocks.{depth}.attn1.to_k.weight"] = k
        converted_state_dict[f"controlnet_transformer_blocks.{depth}.attn1.to_k.bias"] = k_bias
        converted_state_dict[f"controlnet_transformer_blocks.{depth}.attn1.to_v.weight"] = v
        converted_state_dict[f"controlnet_transformer_blocks.{depth}.attn1.to_v.bias"] = v_bias
        # Projection.
        converted_state_dict[f"controlnet_transformer_blocks.{depth}.attn1.to_out.0.weight"] = state_dict.pop(
            f"controlnet.{depth}.copied_block.attn.proj.weight"
        )
        converted_state_dict[f"controlnet_transformer_blocks.{depth}.attn1.to_out.0.bias"] = state_dict.pop(
            f"controlnet.{depth}.copied_block.attn.proj.bias"
        )

        # Feed-forward.
        converted_state_dict[f"controlnet_transformer_blocks.{depth}.ff.net.0.proj.weight"] = state_dict.pop(
            f"controlnet.{depth}.copied_block.mlp.fc1.weight"
        )
        converted_state_dict[f"controlnet_transformer_blocks.{depth}.ff.net.0.proj.bias"] = state_dict.pop(
            f"controlnet.{depth}.copied_block.mlp.fc1.bias"
        )
        converted_state_dict[f"controlnet_transformer_blocks.{depth}.ff.net.2.weight"] = state_dict.pop(
            f"controlnet.{depth}.copied_block.mlp.fc2.weight"
        )
        converted_state_dict[f"controlnet_transformer_blocks.{depth}.ff.net.2.bias"] = state_dict.pop(
            f"controlnet.{depth}.copied_block.mlp.fc2.bias"
        )

        # Cross-attention.
        q = state_dict.pop(f"controlnet.{depth}.copied_block.cross_attn.q_linear.weight")
        q_bias = state_dict.pop(f"controlnet.{depth}.copied_block.cross_attn.q_linear.bias")
        k, v = torch.chunk(state_dict.pop(f"controlnet.{depth}.copied_block.cross_attn.kv_linear.weight"), 2, dim=0)
        k_bias, v_bias = torch.chunk(state_dict.pop(f"controlnet.{depth}.copied_block.cross_attn.kv_linear.bias"), 2, dim=0)

        converted_state_dict[f"controlnet_transformer_blocks.{depth}.attn2.to_q.weight"] = q
        converted_state_dict[f"controlnet_transformer_blocks.{depth}.attn2.to_q.bias"] = q_bias
        converted_state_dict[f"controlnet_transformer_blocks.{depth}.attn2.to_k.weight"] = k
        converted_state_dict[f"controlnet_transformer_blocks.{depth}.attn2.to_k.bias"] = k_bias
        converted_state_dict[f"controlnet_transformer_blocks.{depth}.attn2.to_v.weight"] = v
        converted_state_dict[f"controlnet_transformer_blocks.{depth}.attn2.to_v.bias"] = v_bias

        converted_state_dict[f"controlnet_transformer_blocks.{depth}.attn2.to_out.0.weight"] = state_dict.pop(
            f"controlnet.{depth}.copied_block.cross_attn.proj.weight"
        )
        converted_state_dict[f"controlnet_transformer_blocks.{depth}.attn2.to_out.0.bias"] = state_dict.pop(
            f"controlnet.{depth}.copied_block.cross_attn.proj.bias"
        )

    # DiT XL/2
    transformer = Transformer2DModel(
        sample_size=args.image_size // 8,
        num_layers=28,
        attention_head_dim=72,
        in_channels=4,
        out_channels=8,
        patch_size=2,
        attention_bias=True,
        num_attention_heads=16,
        cross_attention_dim=1152,
        activation_fn="gelu-approximate",
        num_embeds_ada_norm=1000,
        norm_type="ada_norm_single",
        norm_elementwise_affine=False,
        norm_eps=1e-6,
        caption_channels=4096,
    )

    transformer.load_state_dict(converted_state_dict, strict=True)

    num_model_params = sum(p.numel() for p in transformer.parameters())
    print(f"Total number of transformer parameters: {num_model_params}")

    transformer.save_pretrained(os.path.join(args.dump_path, "transformer"))

if __name__ == "__main__":
    parser = argparse.ArgumentParser()

    parser.add_argument("--orig_ckpt_path", default=None, type=str, required=False, help="Path to the checkpoint to convert.")
    parser.add_argument("--dump_path", default=None, type=str, required=True, help="Path to the output pipeline.")
        
    args = parser.parse_args()
    main(args)
