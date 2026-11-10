from pathlib import Path
import hashlib
HERE=Path(__file__).resolve().parent
BASE=Path('/root/wmsr-repro')
def sha(path):return hashlib.sha256(path.read_bytes()).hexdigest()

def accumulated_l1(torch, model, low, high, micro_batch=2):
    """One effective batch, loss normalized over all images, optimizer called by caller once."""
    outputs = []
    total = 0.
    for start in range(0, len(low), micro_batch):
        prediction = model(low[start:start + micro_batch])
        target = high[start:start + micro_batch]
        assert prediction.shape == target.shape
        loss = torch.nn.functional.l1_loss(prediction, target) * len(target) / len(low)
        assert torch.isfinite(loss)
        loss.backward()
        total += loss.item()
        outputs.append(prediction.detach())
    return total, torch.cat(outputs)
