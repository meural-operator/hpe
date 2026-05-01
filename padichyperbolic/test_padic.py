import torch
import sys
import os

# Add paths to sys.path so we can import from adicton and hyperbolic
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..', 'hyperbolic')))
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..', 'adicton')))

from padichyperbolic.model.network import PAdicHyperbolicHPE

def test_padic_hyperbolic():
    print("Testing P-Adic Hyperbolic HPE...")
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"Using device: {device}")
    
    # 243 frames maps to p=3, n=5
    T = 243
    J = 17
    C = 3
    B = 2
    
    model = PAdicHyperbolicHPE(
        in_features=C, 
        embed_dim=128,  # Smaller dim for testing
        num_spatial=2, 
        num_temporal=2,
        p=3, n=5, modes=15, 
        mlp_ratio=4, dropout=0.1
    ).to(device)
    
    x = torch.randn(B, T, J, C, device=device)
    x_vel = torch.randn(B, T, J, C, device=device)
    topo_bias = torch.randn(J, J, device=device)
    
    print(f"Total trainable parameters: {sum(p.numel() for p in model.parameters() if p.requires_grad):,}")
    
    print("Running forward pass...")
    try:
        out = model(x, x_vel, topo_bias)
        print(f"Forward pass successful! Output shape: {out.shape}")
        assert out.shape == (B, T, J, 3)
    except Exception as e:
        print(f"Forward pass failed: {e}")
        return
        
    print("Running backward pass...")
    try:
        loss = out.sum()
        loss.backward()
        print("Backward pass successful!")
    except Exception as e:
        print(f"Backward pass failed: {e}")

if __name__ == "__main__":
    test_padic_hyperbolic()
