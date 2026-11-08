import torch
import torch.nn as nn

class LoRALinear(nn.Module):
    """Frozen linear layer plus (alpha / r) * B(A(x))."""
    def __init__(self, base, r, alpha=None):
        super().__init__()
        if not isinstance(base, nn.Linear):
            raise TypeError("base must be nn.Linear")
        if r < 1:
            raise ValueError("r must be >= 1")
        self.base = base
        self.r = int(r)
        self.alpha = float(2 * r if alpha is None else alpha)
        self.scale = self.alpha / self.r

        for p in self.base.parameters():
            p.requires_grad = False

        self.A = nn.Parameter(torch.empty(self.r, base.in_features, device=base.weight.device, dtype=base.weight.dtype))
        self.B = nn.Parameter(torch.zeros(base.out_features, self.r, device=base.weight.device, dtype=base.weight.dtype))
        nn.init.kaiming_uniform_(self.A, a=5 ** 0.5)

    def forward(self, x):
        return self.base(x) + (x @ self.A.T @ self.B.T) * self.scale

    def merged(self):
        """Return an ordinary nn.Linear with the LoRA update merged into its weight."""
        layer = nn.Linear(
            self.base.in_features, self.base.out_features,
            bias=self.base.bias is not None,
            device=self.base.weight.device, dtype=self.base.weight.dtype
        )
        with torch.no_grad():
            layer.weight.copy_(self.base.weight + self.scale * (self.B @ self.A))
            if self.base.bias is not None:
                layer.bias.copy_(self.base.bias)
        return layer


def replace_qv_with_lora(model, r, alpha=None):
    """Replace every q_proj and v_proj nn.Linear module with the custom LoRA layer."""
    for param in model.parameters():
        param.requires_grad = False
    replaced = []
    def visit(parent, prefix=""):
        for name, child in list(parent.named_children()):
            path = f"{prefix}.{name}" if prefix else name
            if name in {"q_proj", "v_proj"} and isinstance(child, nn.Linear):
                setattr(parent, name, LoRALinear(child, r, alpha))
                replaced.append(path)
            else:
                visit(child, path)
    visit(model)
    if not replaced:
        raise RuntimeError("No q_proj/v_proj nn.Linear modules found.")
    return replaced


def expected_lora_parameters(model, r):
    """Formula: sum over targeted layers of r * (in_features + out_features)."""
    total = 0
    for module in model.modules():
        if isinstance(module, nn.Linear):
            # This helper is intended for an unwrapped model.
            continue
        if isinstance(module, LoRALinear):
            total += r * (module.base.in_features + module.base.out_features)
    return total
