import sys
sys.path.insert(0, 'c:/Users/DIAT/ashish/hpe/hyperbolic_hpe')
import torch
from model.network import HyperbolicHPE

torch.cuda.empty_cache()
torch.cuda.reset_peak_memory_stats()

m = HyperbolicHPE(3, 512, 4, 4, 0.1).cuda()
print(f"Params: {sum(p.numel() for p in m.parameters()):,}")

# Simulate actual training: batch=2, chunk=16 frames, 17 joints
# Total entries per chunk = 2 * 16 = 32
for batch_mult in [32, 64, 128]:
    torch.cuda.empty_cache()
    torch.cuda.reset_peak_memory_stats()
    x = torch.randn(batch_mult, 17, 3).cuda()
    v = torch.randn(batch_mult, 17, 3).cuda() * 0.1
    t = torch.zeros(17, 17).cuda()
    pred = m(x, v, t)
    pred.sum().backward()
    mem = torch.cuda.max_memory_allocated() / 1e9
    print(f"Batch={batch_mult:4d} | VRAM: {mem:.2f} GB | NaN: {torch.isnan(pred).any().item()}")
    del x, v, pred

print("VRAM PROFILE DONE")
