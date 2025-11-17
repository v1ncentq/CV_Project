# utils/eval_utils.py
import torch
import torch.nn.functional as F

def ensemble_predict(models, inputs, device):
    with torch.no_grad():
        logits_sum = None
        for m in models:
            m.eval()
            outputs = m(inputs.to(device))
            if logits_sum is None:
                logits_sum = outputs
            else:
                logits_sum += outputs
        probs = F.softmax(logits_sum / len(models), dim=1)
    return probs
