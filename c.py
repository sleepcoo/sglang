import torch


def apply_rotary_emb(x, cos, sin):
    # x shape: [2, 4]
    # Reshape to process pairs
    x = x.view(2, 2, 2)
    print(x)
    print(x[:, :, 0], x[:, :, 1])
    # Apply rotation to each pair
    rx1 = x[:, :, 0] * cos - x[:, :, 1] * sin
    rx2 = x[:, :, 0] * sin + x[:, :, 1] * cos

    # Combine back
    return torch.stack([rx1, rx2], dim=-1).view(2, 4)


# Input values
q = torch.tensor([0.1602, 0.0684, 0.1172, -2.2656])
cos = torch.tensor([0.5403, 0.6861])
sin = torch.tensor([0.84147, 0.72746])

# Reshape q to match the expected format (2 rows of 4 values)
q = q.repeat(2, 1)  # Duplicate the row to get 2x4

# Apply RoPE
result = apply_rotary_emb(q, cos, sin)

print("Original q:", q)
print("Result after RoPE:", result)
