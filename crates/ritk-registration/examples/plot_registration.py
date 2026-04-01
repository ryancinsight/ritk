import numpy as np
import matplotlib.pyplot as plt
import os

size = (64, 64)

def load_raw(filename):
    if os.path.exists(filename):
        return np.fromfile(filename, dtype=np.float32).reshape(size)
    else:
        return np.zeros(size, dtype=np.float32)

fixed = load_raw("fixed_slice.raw")
moving = load_raw("moving_slice.raw")
moved = load_raw("moved_slice.raw")

threshold = 0.5
def compute_dice(img1, img2):
    i1 = img1 > threshold
    i2 = img2 > threshold
    intersection = np.logical_and(i1, i2).sum()
    union = np.logical_or(i1, i2).sum()
    dice = 2 * intersection / (i1.sum() + i2.sum()) if (i1.sum() + i2.sum()) > 0 else 0
    iou = intersection / union if union > 0 else 0
    return dice, iou

pre_dice, pre_iou = compute_dice(fixed, moving)
post_dice, post_iou = compute_dice(fixed, moved)

def create_overlay(target, source):
    # Target in Green, Source in Magenta
    # Perfect overlap yields White
    rgb = np.zeros((*size, 3), dtype=np.float32)
    rgb[..., 1] = target  # Green component
    rgb[..., 0] = source  # Red component
    rgb[..., 2] = source  # Blue component
    # Clip to max 1.0
    return np.clip(rgb, 0.0, 1.0)

fig, axes = plt.subplots(1, 2, figsize=(12, 6))

axes[0].imshow(create_overlay(fixed, moving), origin='lower')
axes[0].set_title(f'Pre-registration Overlay\nTarget (Green) vs Moving (Magenta)\nDice: {pre_dice:.3f} | IoU: {pre_iou:.3f}')
axes[0].axis('off')

axes[1].imshow(create_overlay(fixed, moved), origin='lower')
axes[1].set_title(f'Post-registration Overlay\nTarget (Green) vs Moved (Magenta)\nDice: {post_dice:.3f} | IoU: {post_iou:.3f}')
axes[1].axis('off')

plt.tight_layout()
plt.savefig("registration_results.png", dpi=150)
print("Saved registration_results.png overlay.")
