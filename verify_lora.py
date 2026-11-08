import torch
import torch.nn as nn
from peft import LoraConfig, get_peft_model
from lora_layer import LoRALinear

class TinyAttention(nn.Module):
    def __init__(self):
        super().__init__()
        self.q_proj = nn.Linear(12, 10)
        self.v_proj = nn.Linear(12, 10)
    def forward(self, x):
        return self.q_proj(x) + self.v_proj(x)

class TinyModel(nn.Module):
    def __init__(self):
        super().__init__()
        self.attn = TinyAttention()
    def forward(self, x):
        return self.attn(x)

def run_checks():
    torch.manual_seed(7)
    x = torch.randn(4, 5, 12)
    base = nn.Linear(12, 10)
    custom = LoRALinear(base, r=3, alpha=6)

    # B=0 makes the adapter exactly identical to the original layer.
    torch.testing.assert_close(custom(x), base(x), rtol=0, atol=0)
    expected = 3 * (12 + 10)
    actual = sum(p.numel() for p in custom.parameters() if p.requires_grad)
    assert actual == expected, (actual, expected)

    with torch.no_grad():
        custom.B.normal_(mean=0.0, std=0.1)
    merged = custom.merged()
    torch.testing.assert_close(custom(x), merged(x), rtol=1e-5, atol=1e-6)

    # Compare the custom layer with PEFT using identical base weights and A/B.
    base2 = nn.Linear(12, 10)
    base2.load_state_dict(base.state_dict())
    class OneProjection(nn.Module):
        def __init__(self, layer):
            super().__init__()
            self.q_proj = layer
        def forward(self, z):
            return self.q_proj(z)
    peft_base = OneProjection(base2)
    peft_model = get_peft_model(
        peft_base,
        LoraConfig(r=3, lora_alpha=6, lora_dropout=0.0,
                   bias="none", target_modules=["q_proj"])
    )
    peft_layer = peft_model.base_model.model.q_proj
    with torch.no_grad():
        peft_layer.lora_A["default"].weight.copy_(custom.A)
        peft_layer.lora_B["default"].weight.copy_(custom.B)
    torch.testing.assert_close(custom(x), peft_model(x), rtol=1e-5, atol=1e-6)

    print("PASS: zero-initialization equality")
    print(f"PASS: trainable parameters = {actual} = r*(in+out)")
    print("PASS: merged layer equals two-path LoRA")
    print("PASS: custom layer agrees with PEFT after copying A and B")

if __name__ == "__main__":
    run_checks()
