"""
Extract Mahjong Soul tile face templates (Original & Cat themes, with/without corner indices)
and ALL tile back templates (mjpai) from the game's StreamingAssets into assets/templates.npz.
"""
import os
import sys
import numpy as np
from PIL import Image
import UnityPy

UnityPy.config.FALLBACK_UNITY_VERSION = '2022.3.62f2c1'

GAME_SA = r'Z:\SteamLibrary\steamapps\common\MahjongSoul\Jantama_MahjongSoul_Data\StreamingAssets\StandaloneWindows'
OUT_DIR = os.path.join(os.path.dirname(os.path.abspath(__file__)), 'assets')
OUT_PATH = os.path.join(OUT_DIR, 'templates.npz')

# Canonical 37 tile names (0m/0p/0s = red fives)
TILE_NAMES = [
    '0m', '1m', '2m', '3m', '4m', '5m', '6m', '7m', '8m', '9m',
    '0p', '1p', '2p', '3p', '4p', '5p', '6p', '7p', '8p', '9p',
    '0s', '1s', '2s', '3s', '4s', '5s', '6s', '7s', '8s', '9s',
    '1z', '2z', '3z', '4z', '5z', '6z', '7z',
]

# Order in mjp.png (512x512, 7 cols x 6 rows, each cell 52x72)
MJP_ORDER = [
    '0m', '1m', '2m', '3m', '4m', '5m', '6m',
    '7m', '8m', '9m', '0p', '1p', '2p', '3p',
    '4p', '5p', '6p', '7p', '8p', '9p', '0s',
    '1s', '2s', '3s', '4s', '5s', '6s', '7s',
    '8s', '9s', '1z', '2z', '3z', '4z', '5z',
    '6z', '7z',
]

# Order in hand.png (800x516, 10 cols x 4 rows, each cell 80x129)
HAND_GRID = {
    '0s': (0, 0), '1s': (1, 0), '2s': (2, 0), '3s': (3, 0), '4s': (4, 0),
    '5s': (5, 0), '6s': (6, 0), '7s': (7, 0), '8s': (8, 0), '9s': (9, 0),
    '0m': (0, 1), '1m': (1, 1), '2m': (2, 1), '3m': (3, 1), '4m': (4, 1),
    '5m': (5, 1), '6m': (6, 1), '7m': (7, 1), '8m': (8, 1), '9m': (9, 1),
    '0p': (0, 2), '1p': (1, 2), '2p': (2, 2), '3p': (3, 2), '4p': (4, 2),
    '5p': (5, 2), '6p': (6, 2), '7p': (7, 2), '8p': (8, 2), '9p': (9, 2),
    '1z': (0, 3), '2z': (1, 3), '3z': (2, 3), '4z': (3, 3), '5z': (4, 3),
    '6z': (5, 3), '7z': (6, 3),
}

# Template sizes
HAND_W, HAND_H = 36, 54
MJP_W, MJP_H = 30, 42
ICON_W, ICON_H = 40, 64


def composite_on_tile_bg(rgba_img: Image.Image, bg_rgb=(232, 232, 229)) -> Image.Image:
    rgba = rgba_img.convert('RGBA')
    bg = Image.new('RGBA', rgba.size, (*bg_rgb, 255))
    return Image.alpha_composite(bg, rgba).convert('RGB')


def extract_mjpface_bundle(bundle_path: str):
    env = UnityPy.load(bundle_path)
    texs = {}
    for o in env.objects:
        if o.type.name == 'Texture2D':
            t = o.read()
            texs[t.m_Name] = t.image
    return texs


def slice_hand_tiles(hand_img: Image.Image, out_w=HAND_W, out_h=HAND_H):
    """Slice 800x516 hand atlas into 37 inner-face RGB templates (cropping off top bevel lip)."""
    arr_out = np.zeros((len(TILE_NAMES), out_h, out_w, 3), dtype=np.uint8)
    rgb = composite_on_tile_bg(hand_img, (233, 233, 230))
    for i, name in enumerate(TILE_NAMES):
        col, row = HAND_GRID[name]
        x0, y0 = col * 80, row * 129
        # Crop inner face (skip top 14px lip and outer 4px rounded edge)
        crop = rgb.crop((x0 + 4, y0 + 15, x0 + 76, y0 + 125))
        resized = crop.resize((out_w, out_h), Image.Resampling.BILINEAR)
        arr_out[i] = np.array(resized, dtype=np.uint8)
    return arr_out


def slice_mjp_tiles(mjp_img: Image.Image, out_w=MJP_W, out_h=MJP_H):
    """Slice 512x512 mjp atlas into 37 RGB templates (52x72 cells)."""
    arr_out = np.zeros((len(TILE_NAMES), out_h, out_w, 3), dtype=np.uint8)
    rgb = mjp_img.convert('RGB')
    pos_map = {name: (idx % 7, idx // 7) for idx, name in enumerate(MJP_ORDER)}
    for i, name in enumerate(TILE_NAMES):
        col, row = pos_map[name]
        x0, y0 = col * 52, row * 72
        crop = rgb.crop((x0 + 2, y0 + 2, x0 + 50, y0 + 70))
        resized = crop.resize((out_w, out_h), Image.Resampling.BILINEAR)
        arr_out[i] = np.array(resized, dtype=np.uint8)
    return arr_out


def slice_ui_icons(hand_img: Image.Image):
    """Slice crisp 40x64 icons from mjpface_default hand.png for GUI display."""
    arr_out = np.zeros((len(TILE_NAMES), ICON_H, ICON_W, 3), dtype=np.uint8)
    rgba = hand_img.convert('RGBA')
    bg = Image.new('RGBA', (80, 129), (30, 34, 42, 255))
    for i, name in enumerate(TILE_NAMES):
        col, row = HAND_GRID[name]
        x0, y0 = col * 80, row * 129
        cell = rgba.crop((x0, y0, x0 + 80, y0 + 129))
        # Composite onto white tile body inside dark UI bg? Actually hand.png already has opaque tile body
        comp = Image.alpha_composite(bg, cell).convert('RGB')
        arr_out[i] = np.array(comp.resize((ICON_W, ICON_H), Image.Resampling.LANCZOS), dtype=np.uint8)
    return arr_out


def main():
    os.makedirs(OUT_DIR, exist_ok=True)
    if not os.path.isdir(GAME_SA):
        raise FileNotFoundError(f'Game StreamingAssets not found: {GAME_SA}')

    face_bundles = {
        'default': None,
        'cat': None,
    }
    mjpai_bundles = []

    for fn in sorted(os.listdir(GAME_SA)):
        fp = os.path.join(GAME_SA, fn)
        if fn.startswith('uwv8ssu@8b625822'):
            env = UnityPy.load(fp)
            for o in env.objects:
                if o.type.name == 'AssetBundle':
                    paths = [c[0] for c in (o.read().m_Container or [])]
                    if any('3d/texture/' in p for p in paths):
                        if any('mjpface_default' in p for p in paths):
                            face_bundles['default'] = fp
                        elif any('mjpface_2308event' in p for p in paths):
                            face_bundles['cat'] = fp
                    break
        elif fn.startswith('uwv8ssu@8b1@002e'):
            env = UnityPy.load(fp)
            for o in env.objects:
                if o.type.name == 'AssetBundle':
                    paths = [c[0] for c in (o.read().m_Container or [])]
                    if any('3d/texture/' in p for p in paths):
                        theme = paths[0].split('/')[3]
                        mjpai_bundles.append((theme, fp))
                    break

    assert face_bundles['default'] and face_bundles['cat'], f'Missing face bundles: {face_bundles}'
    print(f'Found default face, cat face, and {len(mjpai_bundles)} mjpai tile-back bundles.')

    hand_variants = []
    hand_variant_names = []
    mjp_variants = []
    mjp_variant_names = []
    ui_icons = None

    for theme_name in ('default', 'cat'):
        texs = extract_mjpface_bundle(face_bundles[theme_name])
        for suffix, label in [('hand', 'clean'), ('hand_0', 'idx')]:
            if suffix in texs:
                hand_variants.append(slice_hand_tiles(texs[suffix]))
                hand_variant_names.append(f'{theme_name}_{label}')
        for suffix, label in [('mjp', 'clean'), ('mjp_0', 'idx')]:
            if suffix in texs:
                mjp_variants.append(slice_mjp_tiles(texs[suffix]))
                mjp_variant_names.append(f'{theme_name}_{label}')
        if theme_name == 'default' and 'hand' in texs:
            ui_icons = slice_ui_icons(texs['hand'])

    # Extract all 62 tile backs (both 2D hand back and 3D mjp back)
    back_names = []
    back_hand = []
    back_mjp = []

    for theme, fp in mjpai_bundles:
        texs = extract_mjpface_bundle(fp)
        back_names.append(theme)

        # 2D hand.png right half (x=80..160, y=0..129) -> crop inner face matching hand_tiles
        h_img = composite_on_tile_bg(texs['hand'], (233, 233, 230))
        hb_crop = h_img.crop((80 + 4, 15, 80 + 76, 125))
        back_hand.append(np.array(hb_crop.resize((HAND_W, HAND_H), Image.Resampling.BILINEAR), dtype=np.uint8))

        # 3D mjp.png (256x256) -> inner tile back region [29..228, 13..244]
        m_img = texs['mjp'].convert('RGB')
        mb_crop = m_img.crop((32, 16, 224, 240))
        back_mjp.append(np.array(mb_crop.resize((MJP_W, MJP_H), Image.Resampling.BILINEAR), dtype=np.uint8))

    np.savez_compressed(
        OUT_PATH,
        tile_names=np.array(TILE_NAMES),
        hand_variant_names=np.array(hand_variant_names),
        hand_templates=np.stack(hand_variants, axis=0),   # (4, 37, H, W, 3)
        mjp_variant_names=np.array(mjp_variant_names),
        mjp_templates=np.stack(mjp_variants, axis=0),     # (4, 37, H, W, 3)
        back_names=np.array(back_names),
        back_hand=np.stack(back_hand, axis=0),            # (62, H_h, W_h, 3)
        back_mjp=np.stack(back_mjp, axis=0),              # (62, H_m, W_m, 3)
        ui_icons=ui_icons,                                # (37, 64, 40, 3)
    )
    size_kb = os.path.getsize(OUT_PATH) / 1024
    print(f'Saved {OUT_PATH} ({size_kb:.1f} KB)')
    print('  hand_templates:', np.stack(hand_variants, axis=0).shape, hand_variant_names)
    print('  mjp_templates:', np.stack(mjp_variants, axis=0).shape, mjp_variant_names)
    print('  back_hand:', np.stack(back_hand, axis=0).shape)
    print('  back_mjp:', np.stack(back_mjp, axis=0).shape)


if __name__ == '__main__':
    main()
