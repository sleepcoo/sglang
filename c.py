import torch

path = "/root/.cache/huggingface/hub/models--lukeysong--Llama-4-Scout-17B-16E-Eagle3/snapshots/b1d8a533361fcf7cfef112fcea6be9113f18b071/"

# torch.set_printoptions(threshold=float('inf'))
input_embeds_rms = torch.load(path + "draft_outputs.pt")
print("draft_outputs.pt ", input_embeds_rms.shape, input_embeds_rms)


input_embeds_rms = torch.load(path + "attn_out.pt")
print("attn_out.pt ", input_embeds_rms.shape, input_embeds_rms)
