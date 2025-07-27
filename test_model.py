import timm
import torch
import torch.nn as nn

# ---- 1. Load pretrained Swin ----
model = timm.create_model("convnext_tiny", pretrained=True)

print("before", model.head)

num_classes = 10
model.head.fc = nn.Linear(model.head.fc.in_features, num_classes)

print("after", model)


# ---- 4. Test it ----
inputs = torch.randn(64, 3, 224, 224)
outputs = model(inputs)
print(f"Output shape: {outputs.shape}")  # should be [64, 10]
