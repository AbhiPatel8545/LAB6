import torch

def answer_only_labels(input_ids, attention_mask, prompt_lens):
    """
    Score only the answer token that follows each prompt.
    prompt_lens[i] is the index of the answer token in the padded sequence.
    """
    labels = torch.full_like(input_ids, -100)
    rows = torch.arange(input_ids.size(0), device=input_ids.device)
    cols = prompt_lens.to(input_ids.device)
    labels[rows, cols] = input_ids[rows, cols]

    # Every example contributes exactly one target, and no prompt position is scored.
    assert torch.all((labels != -100).sum(dim=1) == 1)
    assert torch.all(labels[rows, cols] == input_ids[rows, cols])
    assert torch.all(cols < attention_mask.sum(dim=1).to(cols.device))
    return labels
