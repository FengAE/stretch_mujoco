"""Generate differentiated NPC texture variants from the base SMPL diffuse map.

Creates three distinct employee appearances by adjusting skin tone and clothing
colour in the SMPL-X UV texture. Each variant keeps the original mesh/UV layout
so the same animation frame OBJs can be reused.
"""

from __future__ import annotations

from pathlib import Path

import cv2
import numpy as np

PROJECT_ROOT = Path(__file__).resolve().parents[1]
MODELS_ROOT = PROJECT_ROOT / "stretch_mujoco" / "models"
HUMANOID_ROOT = MODELS_ROOT / "assets" / "humanoid" / "generated" / "animations"
BASE_TEXTURE = HUMANOID_ROOT / "smplx_employee_diffuse.png"
OUTPUT_DIR = HUMANOID_ROOT

NPC_VARIANTS = {
    "npc_01": {
        "label": "Employee 01 – Warm skin, navy suit",
        "skin_brightness": 0.88,  # slightly darker
        "cloth_hue_shift": 0,     # keep blue tones
        "cloth_saturation": 1.25,  # richer navy
        "overall_warmth": 0.08,    # warmer white balance
    },
    "npc_02": {
        "label": "Employee 02 – Lighter skin, charcoal grey suit",
        "skin_brightness": 1.10,   # lighter
        "cloth_hue_shift": 0,      # neutral
        "cloth_saturation": 0.55,   # desaturated grey
        "overall_warmth": -0.04,    # cooler white balance
    },
    "npc_03": {
        "label": "Employee 03 – Medium skin, brown jacket",
        "skin_brightness": 0.97,   # medium
        "cloth_hue_shift": 15,      # shift blues toward brown
        "cloth_saturation": 1.25,   # richer brown
        "overall_warmth": 0.04,     # slightly warmer
    },
}


def _load_texture(path: Path) -> np.ndarray:
    """Read texture preserving the alpha channel if present."""
    image = cv2.imread(str(path), cv2.IMREAD_UNCHANGED)
    if image is None:
        raise FileNotFoundError(f"Failed to load texture: {path}")
    return image


def _apply_variant(source: np.ndarray, params: dict) -> np.ndarray:
    """Create a single NPC texture variant with global colour adjustments.

    The SMPL-X UV map packs face, hands, and clothing into a single atlas.
    Rather than attempt precise region segmentation (which depends on the
    specific UV layout), we apply artistically motivated global transforms
    that produce clearly distinct appearances.
    """
    # Work in float32 for precision
    img = source.astype(np.float32) / 255.0

    has_alpha = img.shape[-1] == 4
    if has_alpha:
        alpha = img[:, :, 3:4]
        img = img[:, :, :3]
    else:
        alpha = None

    # --- Skin tone adjustment ---
    # Skin occupies warm hues (reddish) in the texture. We shift skin by
    # adjusting brightness in the luminance channel.
    lab = cv2.cvtColor(img, cv2.COLOR_BGR2LAB).astype(np.float32)
    # Lightness channel (L): [0, 100]
    l_channel = lab[:, :, 0]
    skin_brightness = params["skin_brightness"]
    # Skin tones typically sit in the mid-range of L. Apply a global L
    # multiplier biased toward the centre of the range.
    l_mean = np.mean(l_channel)
    l_channel = np.clip(l_channel * skin_brightness, 0, 100)
    lab[:, :, 0] = l_channel
    img = cv2.cvtColor(lab.astype(np.uint8), cv2.COLOR_LAB2BGR).astype(np.float32) / 255.0

    # --- Clothing colour adjustment ---
    # Clothing occupies cooler hues (blues, greys). We shift hue and
    # adjust saturation for distinct outfits.
    hsv = cv2.cvtColor(img.astype(np.float32), cv2.COLOR_BGR2HSV)
    # HSV ranges in OpenCV: H [0, 180), S [0, 255], V [0, 255]
    h, s, v = hsv[:, :, 0], hsv[:, :, 1], hsv[:, :, 2]

    # Target clothing hues: blues through cyans (H ~ 80-140)
    cloth_mask = (h > 70) & (h < 150) & (s > 15)
    h[cloth_mask] = np.clip(
        h[cloth_mask] + params["cloth_hue_shift"], 0, 179
    )
    s[cloth_mask] = np.clip(s[cloth_mask] * params["cloth_saturation"], 0, 255)

    hsv[:, :, 0] = h
    hsv[:, :, 1] = s
    img = cv2.cvtColor(hsv, cv2.COLOR_HSV2BGR)

    # --- Overall white-balance warmth ---
    warmth = params["overall_warmth"]
    if warmth != 0:
        # Shift red channel slightly for warm/cool
        img[:, :, 2] = np.clip(img[:, :, 2] + warmth, 0, 1)  # red
        img[:, :, 0] = np.clip(img[:, :, 0] - warmth, 0, 1)  # blue

    # Convert back to uint8
    img = np.clip(img * 255, 0, 255).astype(np.uint8)

    if has_alpha:
        img = np.dstack([img, (alpha * 255).astype(np.uint8)])

    return img


def main() -> None:
    if not BASE_TEXTURE.exists():
        raise SystemExit(f"Base texture not found: {BASE_TEXTURE}")

    source = _load_texture(BASE_TEXTURE)
    print(f"Loaded base texture: {BASE_TEXTURE}")
    print(f"  Size: {source.shape[1]}x{source.shape[0]}, channels: {source.shape[-1]}")

    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

    for variant_id, params in NPC_VARIANTS.items():
        output_path = OUTPUT_DIR / f"smplx_{variant_id}_diffuse.png"
        variant = _apply_variant(source, params)
        cv2.imwrite(str(output_path), variant)
        print(f"  -> {output_path.name}  ({params['label']})")

    print(f"\nGenerated {len(NPC_VARIANTS)} NPC texture variants in {OUTPUT_DIR}")


if __name__ == "__main__":
    main()
