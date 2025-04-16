# Save dummy ManTraNet model (PyTorch)
import torch

dummy_model = torch.nn.Conv2d(3, 1, kernel_size=3, padding=1)
torch.save(dummy_model.state_dict(), "mantranet.pth")
print("✅ Saved: mantranet.pth")
