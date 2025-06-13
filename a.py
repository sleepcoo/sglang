import torch

path = "/root/.cache/huggingface/hub/models--lukeysong--Llama-4-Scout-17B-16E-Eagle3/snapshots/fe6c99094bf16dcc74923b1d22084b6eac1303f3/draft_outputs.pt"


path = "/tmp/torchtune/llama4_17Bx16E/draft/input_embeds_rms.pt"
input_embeds_rms = torch.load(path)
print("input_embeds_rms ", input_embeds_rms.shape, input_embeds_rms)

path = "/tmp/torchtune/llama4_17Bx16E/draft/fused_features_rms.pt"
fused_features_rms = torch.load(path)
print("fused_features_rms ", fused_features_rms.shape, fused_features_rms)

path = "/tmp/torchtune/llama4_17Bx16E/draft/h.pt"
attn_input = torch.load(path)
print("attn input ", attn_input.shape, attn_input)


path = "/tmp/torchtune/llama4_17Bx16E/draft/combined_input.pt"
combined_input = torch.load(path)
print("combined_input ", combined_input.shape, combined_input)

path = "/tmp/torchtune/llama4_17Bx16E/draft/attn_out.pt"
attn = torch.load(path)
print("attn ", attn.shape, attn)


path = "/tmp/torchtune/llama4_17Bx16E/draft/mlp_out.pt"
mlp = torch.load(path)
print("mlp ", mlp.shape, mlp)
