#!/usr/bin/env python3
# render_level.py
"""
Kubric scene composition script for counting data generation (Levels 1-5).

Level definitions:
  Level 1: one asset from one category (single type)
  Level 2: same asset, two sizes or two colors (dual type)
  Level 3: one asset from each of two categories, with variable size (dual type)
  Level 4: two different assets from the same category, with variable size (dual type)
           [training filters out categories with count <= 50]
  Level 5: two categories within the same super category, each with multiple assets (dual type)
           - sample only from super categories containing >=2 categories
           - skip "Structures"

Data splits:
  objects_split ∈ {train, testA, testB}
    - testB: novel category OOD (SUPER_CATEGORIES[*]['test'])
    - testA: novel asset OOD, holding out 1/11 assets per train category with a fixed seed
    - train: train categories excluding testA held-out assets

Additional features:
  - testA holdout: ensure every train category keeps >=2 assets in both train and testA
  - category sampling weights: count < threshold -> weight *= low_count_weight
  - level5 super-category weights: if every available category is below threshold,
    weight *= low_count_weight
  - level2 color: use a predefined color palette and export group-level colors
  - count sampling fix: dual-type levels use >=10 objects per group and 20-250 total objects
  - level5 size ratio fix: if the two groups' base size ratio exceeds M,
    scale the smaller-object group up to big/M

"""

import logging
import sys
import time
import signal
import os
import json
import bpy
import kubric as kb
from kubric.simulator import PyBullet
from kubric.renderer import Blender
import numpy as np
import colorsys

# --------------------------------------------------------------------------
# RNG compatibility: Kubric may return np.random.RandomState (legacy) or
# np.random.Generator (new). Provide a unified interface.
# --------------------------------------------------------------------------
class RNG:
    def __init__(self, rng):
        self.rng = rng

    def randint(self, low, high=None):
        # returns int in [low, high) if high is not None, else [0, low)
        if hasattr(self.rng, "integers"):
            return int(self.rng.integers(low, high))
        return int(self.rng.randint(low, high))

    def uniform(self, low=0.0, high=1.0, size=None):
        if hasattr(self.rng, "uniform"):
            return self.rng.uniform(low, high, size=size)
        # fallback shouldn't happen
        return np.random.uniform(low, high, size=size)

    def random(self):
        if hasattr(self.rng, "random"):
            return float(self.rng.random())
        return float(self.rng.rand())

    def choice(self, a, size=None, replace=True, p=None):
        # RandomState.choice supports p, Generator.choice supports p too.
        return self.rng.choice(a, size=size, replace=replace, p=p)

    def shuffle(self, x):
        self.rng.shuffle(x)

    def rand(self):
        return self.random()

def compute_bbox_and_filter_border_fragments(segmentation, assets,
                                             ref_percentile=50.0,
                                             small_ratio=0.1,
                                             touch_eps_px=1.0):
    """
    Compute per-asset bbox/area from segmentation indices (id=k <-> assets[k-1]),
    then mark border fragments as invisible by overwriting asset.metadata['visibility'].

    Filtering rule (global, not category-specific):
      - touches border (bbox near edge, using eps=touch_eps_px pixels)
      - area < percentile(area_all, ref_percentile) * small_ratio

    Args:
      segmentation: data_stack["segmentation"], shape [T,H,W,1] or [T,H,W] depending on kubric version
      assets: scene.assets (ordering corresponds to segmentation indices starting from 1)
    """
    # normalize segmentation to [T,H,W] integer
    seg = segmentation
    if seg.ndim == 4:
        seg = seg[..., 0]
    T, H, W = seg.shape

    eps = float(touch_eps_px) / float(max(H, W))  # pixel -> normalized

    # We'll compute based on frame 0 (you render single frame anyway)
    seg0 = seg[0]

    # Precompute areas for all ids that appear (for percentile reference)
    present_ids = np.unique(seg0)
    present_ids = present_ids[present_ids > 0]
    present_ids = present_ids.astype(np.int64)

    id_to_bbox = {}
    id_to_area = {}

    for sid in present_ids.tolist():
        ys, xs = np.where(seg0 == sid)
        if ys.size == 0:
            continue
        y0 = float(ys.min() / H)
        x0 = float(xs.min() / W)
        y1 = float((ys.max() + 1) / H)
        x1 = float((xs.max() + 1) / W)
        id_to_bbox[int(sid)] = (y0, x0, y1, x1)
        id_to_area[int(sid)] = max(0.0, (y1 - y0)) * max(0.0, (x1 - x0))

    areas = np.asarray([a for a in id_to_area.values() if a > 0], dtype=np.float64)
    if areas.size == 0:
        logging.info("Border-fragment filter: no valid areas found; skipping.")
        return {"dropped": 0, "total_present": 0}

    ref_area = float(np.percentile(areas, float(ref_percentile)))
    thresh = float(ref_area * float(small_ratio))

    def touches_border(b):
        y0, x0, y1, x1 = b
        return (x0 <= eps) or (y0 <= eps) or (x1 >= 1.0 - eps) or (y1 >= 1.0 - eps)

    dropped = 0
    touch_cnt = 0
    small_cnt = 0
    both_cnt = 0

    # Make sure every asset has visibility field; if compute_visibility already ran, we overwrite selectively.
    for sid, b in id_to_bbox.items():
        if sid < 1 or sid > len(assets):
            continue
        a = float(id_to_area.get(sid, 0.0))
        t = touches_border(b)
        s = a < thresh

        if t:
            touch_cnt += 1
        if s:
            small_cnt += 1
        if t and s:
            both_cnt += 1
            dropped += 1
            # Mark as invisible in all frames (so downstream logic treats it as removed)
            assets[sid - 1].metadata["visibility"] = [0 for _ in range(T)]
        else:
            # Optional: store bbox/area for debugging/metadata
            assets[sid - 1].metadata["bbox_from_seg"] = [b]  # normalized bbox for frame0
            assets[sid - 1].metadata["bbox_area_from_seg"] = a

    logging.info(
        f"Border-fragment filter (global): dropped {dropped} / {len(present_ids)} present instances "
        f"(touch={touch_cnt}, small={small_cnt}, both={both_cnt}, "
        f"ref_pctl={ref_percentile}, ratio={small_ratio}, ref_area={ref_area:.6g}, thresh={thresh:.6g}, eps={eps:.6f})"
    )

    return {"dropped": int(dropped), "total_present": int(len(present_ids)), "ref_area": ref_area, "thresh": thresh}


# ============================================================================
# Timeout handling
# ============================================================================
class TimeoutError(Exception):
    pass

class SlowPlacementError(Exception):
    """Raised when object placement is too slow."""
    pass

SCRIPT_START_TIME = time.time()
TIMEOUT_SECONDS = 30 * 60

def check_timeout():
    elapsed = time.time() - SCRIPT_START_TIME
    if elapsed > TIMEOUT_SECONDS:
        raise TimeoutError(f"Script timeout after {elapsed/60:.1f} minutes")

def timeout_handler(signum, frame):
    raise TimeoutError("Script timeout (SIGALRM)")

signal.signal(signal.SIGALRM, timeout_handler)

logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s [%(levelname)s] %(message)s',
    datefmt='%H:%M:%S',
    stream=sys.stderr
)

bpy.app.debug_wm = False
bpy.app.debug = False

# ============================================================================
# Super-category definitions for Level 5 category selection and the testB split.
# ============================================================================
SUPER_CATEGORIES = {
    # ================= Vehicles =================
    "Vehicles_Water": {
        "train": ["vessel", "boat"],
        "test": []
    },
    "Vehicles_Land_Large": {
        "train": ["airplane", "car", "bus", "truck"],
        "test": ["train", "rocket"]
    },
    "Vehicles_Land_Small": {
        "train": ["motorcycle", "bicycle", "skateboard"],
        "test": []
    },

    # ================= Animals =================
    "Animals_Water": {
        "train": ["fish", "octopus_squid", "crab", "lobster_shrimp"],
        "test": ["turtle", "frog"]
    },
    "Animals_Land_Large": {
        "train": ["cat", "dog", "horse", "deer", "cow", "sheep", "person"],
        "test": ["elephant", "bear"]
    },
    "Animals_Land_Small": {
        "train": ["bird", "butterfly", "beetle"],
        "test": ["lizard", "snake"]
    },

    # ================= Food =================
    "Food_Produce": {
        "train": [
            "banana", "grape", "apple", "strawberry", "tomatoes", "orange",
            "potatoes", "carrot", "onion", "lemon_lime", "cucumber", "eggplant",
            "pepper_vegetable", "broccoli_cauliflower", "radish", "garlic",
            "corn", "watermelon", "pineapple", "cherry"
        ],
        "test": ["pear", "avocado", "pumpkin_squash", "peach"]
    },
    "Food_Processed": {
        "train": [
            "burger", "donut", "sandwich", "baguette", "bread_loaf",
            "croissant", "muffin", "bagel", "pretzel", "candy", "egg",
            "sushi", "ice_cream"
        ],
        "test": ["cake", "pizza"]
    },

    # ================= Furniture and large appliances =================
    "Furniture_Large": {
        "train": [
            "table", "chair", "sofa", "bench", "cabinet", "bookshelf",
            "bed", "piano", "file", "bathtub", "dishwasher"
        ],
        "test": ["stove", "washer"]
    },

    # ================= Household and small objects =================
    "Household_Electronics": {
        "train": [
            "laptop", "computer keyboard", "microwave", "telephone",
            "cellular telephone", "loudspeaker", "camera", "remote control",
            "earphone", "display"
        ],
        "test": ["printer", "microphone"]
    },
    "Weapons_Instruments": {
        "train": ["rifle", "pistol", "knife"],
        "test": ["guitar", "bat"]
    },
    "Household_Containers": {
        "train": [
            "pot", "jar", "bottle", "mug", "bowl", "can", "cup", "plate",
            "bag", "mailbox", "birdhouse"
        ],
        "test": ["ashcan", "basket"]
    },
    "Household_Wearables": {
        "train": [
            "shoe", "T-shirt", "trousers", "glasses", "cap",
            "backpack", "tie", "sock", "helmet"
        ],
        "test": ["hat", "glove"]
    },
    "Household_Hardware_Tools": {
        "train": [
            "faucet", "lamp", "hammer", "pliers", "screwdriver",
            "wrench", "saw", "nail", "screw", "paper_clip",
            "tape_roll", "battery"
        ],
        "test": ["pillow", "clock"]
    },
    "Household_Toys_Misc": {
        "train": [
            "teddy_bear", "ball", "lego_brick", "coin", "bottle_cap",
            "bead", "button", "toilet_paper", "pencil", "fork", "key",
            "spoon", "candle", "matchstick", "cigarette"
        ],
        "test": ["dice", "playing_card", "book"]
    },

    # ================= Special categories =================
    "Structures": {
        "train": ["tower"],
        "test": []
    }
}

def get_train_categories():
    cats = []
    for _, d in SUPER_CATEGORIES.items():
        cats.extend(d.get("train", []))
    return list(dict.fromkeys(cats))

def get_testB_categories():
    cats = []
    for _, d in SUPER_CATEGORIES.items():
        cats.extend(d.get("test", []))
    return list(dict.fromkeys(cats))

def get_categories_in_super_category(super_category, split_key):
    if super_category not in SUPER_CATEGORIES:
        return []
    return SUPER_CATEGORIES[super_category].get(split_key, [])

# ============================================================================
# Category counts table used for sampling weights.
# ============================================================================
CATEGORY_COUNTS_RAW = r"""
Category  Count
               table   8436
               chair   6718
            airplane   4044
                 car   3486
                sofa   3172
               rifle   2373
                lamp   2318
              vessel   1935
               bench   1811
         loudspeaker   1597
             cabinet   1571
             display   1093
           telephone   1088
                 bus    937
             bathtub    856
  cellular telephone    831
              guitar    797
              faucet    744
               clock    650
                 pot    601
                 jar    596
              bottle    498
              laptop    460
           bookshelf    451
               knife    424
               train    389
              ashcan    343
          motorcycle    337
              pistol    307
                file    298
                bird    261
               piano    239
                 bed    233
                shoe    226
               stove    218
                 mug    214
             glasses    190
                bowl    186
             T-shirt    178
            backpack    170
              washer    169
             printer    166
              helmet    162
          skateboard    152
           microwave    152
            trousers    145
               tower    133
                 bat    132
                book    127
               truck    126
                 cup    124
                 dog    113
              camera    113
              basket    113
                fish    113
                 can    108
              person    106
               spoon    104
                cake     96
              pillow     96
               plate     95
             mailbox     94
          dishwasher     92
                boat     88
              rocket     85
                 bag     83
               pizza     75
            earphone     73
           birdhouse     73
               horse     69
          microphone     67
      remote control     66
   computer keyboard     65
                ball     65
                 cat     65
                deer     65
                coin     62
             bicycle     58
              pencil     57
                 cap     56
            elephant     55
                bear     54
              banana     50
                 tie     50
                bead     50
               sheep     47
                fork     45
                 cow     45
              beetle     45
              button     45
               donut     45
                 hat     44
           butterfly     40
               sushi     40
          bottle_cap     40
              burger     36
                 egg     35
               glove     35
          teddy_bear     34
               apple     32
               snake     30
              lizard     30
       octopus_squid     30
           ice_cream     30
                dice     30
               candy     30
    pepper_vegetable     30
            sandwich     27
              candle     25
              turtle     25
                sock     25
          lemon_lime     25
                frog     25
              carrot     25
               onion     25
               screw     24
               grape     24
broccoli_cauliflower     20
            baguette     20
          bread_loaf     20
            cucumber     20
           cigarette     20
                crab     20
                corn     20
           croissant     20
      lobster_shrimp     20
          matchstick     20
               peach     20
                pear     20
      pumpkin_squash     20
        playing_card     20
            tomatoes     20
          strawberry     20
              garlic     20
              radish     20
              orange     20
              muffin     20
            potatoes     18
                nail     18
               bagel     18
          paper_clip     17
             pretzel     15
          watermelon     15
              cherry     15
                 key     15
           tape_roll     15
          lego_brick     15
            eggplant     15
           pineapple     15
        toilet_paper     14
             avocado     14
             battery     12
              hammer     12
              pliers     12
         screwdriver     12
              wrench     12
                 saw      9
"""

def parse_category_counts(raw_text):
    counts = {}
    for line in raw_text.splitlines():
        line = line.rstrip("\n")
        if not line.strip():
            continue
        if line.strip().lower().startswith("category"):
            continue
        if line.strip().lower().startswith("total categories"):
            break
        # robust split: count is last token, category is the rest
        parts = line.strip().split()
        if len(parts) < 2:
            continue
        try:
            cnt = int(parts[-1])
        except ValueError:
            continue
        cat = " ".join(parts[:-1])
        counts[cat] = cnt
    return counts

CATEGORY_COUNTS = parse_category_counts(CATEGORY_COUNTS_RAW)

def get_category_count(category):
    return int(CATEGORY_COUNTS.get(category, 0))

# ============================================================================
# Level2 color palette (predefined colors).
# ============================================================================
# Each entry: (name, (r,g,b)) where rgb in [0,1]
LEVEL2_COLOR_PALETTE = [
    ("red", (0.90, 0.10, 0.10)),
    ("deep red", (0.75, 0.05, 0.05)),
    ("dark red", (0.55, 0.05, 0.05)),

    ("green", (0.10, 0.75, 0.15)),
    ("deep green", (0.05, 0.55, 0.10)),
    ("mint", (0.15, 0.90, 0.35)),

    ("blue", (0.10, 0.30, 0.90)),
    ("deep blue", (0.05, 0.20, 0.65)),
    ("sky blue", (0.20, 0.55, 0.95)),

    ("yellow", (0.95, 0.80, 0.10)),
    ("gold", (0.90, 0.65, 0.05)),
    ("mustard", (0.75, 0.55, 0.10)),

    ("orange", (0.95, 0.45, 0.10)),
    ("deep orange", (0.85, 0.30, 0.05)),
    ("burnt orange", (0.70, 0.25, 0.10)),

    ("purple", (0.70, 0.10, 0.85)),
    ("deep purple", (0.50, 0.05, 0.65)),
    ("lavender", (0.80, 0.35, 0.95)),

    ("pink", (0.95, 0.20, 0.55)),
    ("deep pink", (0.80, 0.10, 0.40)),
    ("light pink", (0.95, 0.55, 0.75)),

    ("brown", (0.55, 0.30, 0.10)),
    ("dark brown", (0.45, 0.25, 0.12)),
    ("tan", (0.65, 0.45, 0.25)),

    ("dark gray", (0.15, 0.15, 0.15)),
    ("gray", (0.30, 0.30, 0.30)),
    ("light gray", (0.55, 0.55, 0.55)),

    ("white", (0.90, 0.90, 0.90)),
    ("bright white", (0.98, 0.98, 0.98)),
    ("black", (0.05, 0.05, 0.05)),

    ("cyan", (0.10, 0.85, 0.85)),
    ("deep cyan", (0.05, 0.65, 0.65)),
    ("aqua", (0.25, 0.95, 0.80)),

    ("crimson", (0.85, 0.10, 0.15)),
    ("lime", (0.10, 0.85, 0.10)),
    ("royal blue", (0.10, 0.10, 0.85)),

    ("lemon", (0.85, 0.85, 0.10)),
    ("magenta", (0.85, 0.10, 0.85)),
    ("turquoise", (0.10, 0.85, 0.85)),
]


def color_distance(rgb1, rgb2):
    return float(np.sqrt((rgb1[0]-rgb2[0])**2 + (rgb1[1]-rgb2[1])**2 + (rgb1[2]-rgb2[2])**2))


def pick_two_distinct_palette_colors(rng, min_dist=0.35, max_tries=200):
    if len(LEVEL2_COLOR_PALETTE) < 2:
        return ("red", (1.0, 0.0, 0.0)), ("green", (0.0, 1.0, 0.0))

    for _ in range(max_tries):
        i = int(rng.randint(0, len(LEVEL2_COLOR_PALETTE)))
        j = int(rng.randint(0, len(LEVEL2_COLOR_PALETTE)))
        if i == j:
            continue
        n1, c1 = LEVEL2_COLOR_PALETTE[i]
        n2, c2 = LEVEL2_COLOR_PALETTE[j]
        if color_distance(c1, c2) >= min_dist:
            return (n1, c1), (n2, c2)

    # fallback: farthest pair
    best = (LEVEL2_COLOR_PALETTE[0], LEVEL2_COLOR_PALETTE[1], -1.0)
    for i in range(len(LEVEL2_COLOR_PALETTE)):
        for j in range(i + 1, len(LEVEL2_COLOR_PALETTE)):
            d = color_distance(LEVEL2_COLOR_PALETTE[i][1], LEVEL2_COLOR_PALETTE[j][1])
            if d > best[2]:
                best = (LEVEL2_COLOR_PALETTE[i], LEVEL2_COLOR_PALETTE[j], d)

    return best[0], best[1]


# ============================================================================
# Hyperparameter config system
# ============================================================================
DEFAULT_CONFIG = {
    "min_objects_per_group": (10, 10),
    "max_total_objects": (250, 250),
    "object_size_min": (0.35, 0.35),
    "object_size_max": (0.85, 0.85),
    "size_variation_min": (0.8, 0.8),
    "size_variation_max": (1.2, 1.2),
    "level2_small_ratio_min": (0.5, 0.5),
    "level2_small_ratio_max": (0.9, 0.9),
    "level2_large_ratio_min": (1.1, 1.1),
    "level2_large_ratio_max": (1.5, 1.5),
    "level2_min_hue_diff": (0.3, 0.3),
    "density_factor": (1.15, 1.15),
    "min_distance_ratio": (0.85, 0.85),
    "placement_attempts": (50, 50),
    "camera_distance_min": (2.5, 2.5),
    "camera_distance_max": (18.0, 18.0),
    "camera_height_min": (1.0, 1.0),
    "camera_height_max": (15.0, 15.0),
    "camera_angle_min": (10.0, 10.0),
    "camera_angle_max": (75.0, 75.0),
    "focal_length_min": (20.0, 20.0),
    "focal_length_max": (65.0, 65.0),
    "coverage_min": (0.2, 0.2),
    "coverage_max": (0.75, 0.75),
}

CATEGORY_CONFIGS = {
    "default": [DEFAULT_CONFIG],
}

def load_config_from_file(config_path):
    if config_path and os.path.exists(config_path):
        with open(config_path, 'r') as f:
            return json.load(f)
    return None

def get_config_for_category(category, external_config=None):
    if external_config:
        if category in external_config.get("category_configs", {}):
            config_list = external_config["category_configs"][category]
        elif "default" in external_config.get("category_configs", {}):
            config_list = external_config["category_configs"]["default"]
        else:
            config_list = [DEFAULT_CONFIG]
    else:
        config_list = CATEGORY_CONFIGS.get(category, CATEGORY_CONFIGS["default"])
    return config_list

def sample_config(config_list, rng):
    if not config_list:
        config_list = [DEFAULT_CONFIG]
    if len(config_list) > 1:
        idx = int(rng.randint(0, len(config_list)))
        selected_config = config_list[idx]
    else:
        selected_config = config_list[0]

    def sample_value(value):
        if isinstance(value, (list, tuple)) and len(value) == 2:
            try:
                min_val, max_val = float(value[0]), float(value[1])
                return float(rng.uniform(min_val, max_val))
            except (TypeError, ValueError):
                return value
        return value

    sampled = {}
    for key, value in DEFAULT_CONFIG.items():
        sampled[key] = sample_value(value)
    for key, value in selected_config.items():
        sampled[key] = sample_value(value)
    return sampled

def to_serializable(obj):
    if isinstance(obj, np.ndarray):
        return obj.tolist()
    elif isinstance(obj, (np.float32, np.float64)):
        return float(obj)
    elif isinstance(obj, (np.int32, np.int64)):
        return int(obj)
    elif isinstance(obj, dict):
        return {k: to_serializable(v) for k, v in obj.items()}
    elif isinstance(obj, (list, tuple)):
        return [to_serializable(item) for item in obj]
    return obj

# ============================================================================
# CLI arguments
# ============================================================================
parser = kb.ArgumentParser()

# objects split: train / testA / testB
parser.add_argument("--objects_split", choices=["train", "testA", "testB"], default="train",
                    help="Object split: train / testA (novel asset) / testB (novel category)")

# HDRI split: keep train/test.
parser.add_argument("--backgrounds_split", choices=["train", "test"], default="train",
                    help="HDRI background split: train/test (10:1)")

# Level selection (1-5)
parser.add_argument("--level", type=int, choices=[1, 2, 3, 4, 5], default=1,
                    help="Generation difficulty level: 1..5")

# Level 2 sub-mode
parser.add_argument("--level2_mode", choices=["size", "color", "random"], default="random",
                    help="Level 2 distinction mode")

# Level 5 parameters (formerly level4)
parser.add_argument("--level5_min_assets_per_category", type=int, default=3)
parser.add_argument("--level5_max_assets_per_category", type=int, default=10)
parser.add_argument("--level5_max_total_objects", type=int, default=250,
                    help="Maximum total objects across the two Level 5 categories")
parser.add_argument("--level5_max_size_ratio", type=float, default=3.0,
                    help="Maximum Level 5 base-size ratio between the two groups (big/small). "
                         "If exceeded, the smaller objects are scaled up.")

# External config file path
parser.add_argument("--config_file", type=str, default="",
                    help="Path to an external hyperparameter config file in JSON format")

# testA heldout fraction
parser.add_argument("--testA_fraction", type=float, default=(1.0/11.0),
                    help="testA heldout fraction for assets within train categories")

parser.add_argument("--testA_split_seed", type=int, default=42,
                    help="Random seed to generate deterministic per-category testA asset split")

# category weighting
parser.add_argument("--low_count_threshold", type=int, default=150,
                    help="If category asset count < threshold, its sampling weight is multiplied by low_count_weight")
parser.add_argument("--low_count_weight", type=float, default=0.5,
                    help="Weight multiplier for low-count categories/super-categories")

# level4 training filter
parser.add_argument("--level4_min_count_for_train", type=int, default=51,
                    help="For Level 4 training only: require category count >= this value (default: 51 means exclude <=50)")

# dual-type minimum per group (hard constraint)
parser.add_argument("--min_objects_per_group_hard", type=int, default=10,
                    help="Hard minimum number of objects per group for dual-type levels (2-5)")

# Floor parameters
parser.add_argument("--floor_friction", type=float, default=0.3)
parser.add_argument("--floor_restitution", type=float, default=0.5)

# Placement parameters
parser.add_argument("--density_factor", type=float, default=1.15)
parser.add_argument("--min_distance_ratio", type=float, default=0.85)
parser.add_argument("--placement_attempts", type=int, default=50)

# Placement speed check parameters
parser.add_argument("--placement_speed_check_count", type=int, default=20)
parser.add_argument("--placement_speed_threshold", type=float, default=0.2)

# Camera parameters
parser.add_argument("--camera_offset_ratio", type=float, default=0.4)

# Timeout
parser.add_argument("--timeout_minutes", type=int, default=30)

# Asset paths
parser.add_argument("--kubasic_assets", type=str, default="assets/KuBasic.json")
parser.add_argument("--hdri_assets", type=str, default="assets/HDRI_haven.json")
parser.add_argument("--hdri_t2l_assets", type=str, default="assets/HDRI_t2l.json")
parser.add_argument("--shapenet_assets", type=str, default="assets/ShapeNetCore.v2.json")
parser.add_argument("--trellis_assets", type=str,
                    default="assets/trellis/metadata.json")
parser.add_argument("--save_state", dest="save_state", action="store_true")

parser.set_defaults(save_state=False, frame_end=5, frame_start=5, frame_rate=1, resolution=512)
FLAGS = parser.parse_args()

TIMEOUT_SECONDS = FLAGS.timeout_minutes * 60
signal.alarm(TIMEOUT_SECONDS)
resolution = int(FLAGS.resolution)

# ============================================================================
# Utility functions
# ============================================================================
def compute_spawn_region(num_objects, avg_obj_size, density_factor=1.2):
    obj_area = (avg_obj_size * density_factor) ** 2
    total_area = num_objects * obj_area
    side_length = np.sqrt(total_area)
    side_length = np.clip(side_length, 2.0, 15.0)
    half = side_length / 2
    logging.info(f"  Dynamic region: {side_length:.2f}x{side_length:.2f} (area: {total_area:.1f})")
    return [(-half, -half, 0.5), (half, half, 4)], side_length

def compute_safe_scale(obj, target_size):
    bounds = obj.bounds
    dimensions = bounds[1] - bounds[0]
    max_dim = np.max(dimensions)
    if max_dim <= 0 or max_dim > 10000:
        return 1.0, max_dim
    scale_factor = target_size / max_dim
    final_size = np.clip(max_dim * scale_factor, 0.1, 3.0)
    return final_size / max_dim, max_dim

def compute_compact_positions(num_objects, avg_obj_size, spawn_region, rng,
                             min_distance_ratio=0.85, max_attempts=50):
    min_x, min_y, _ = spawn_region[0]
    max_x, max_y, _ = spawn_region[1]

    min_distance = avg_obj_size * min_distance_ratio
    positions = []
    margin = avg_obj_size * 0.3

    cell_size = min_distance * 1.2
    grid = {}

    def get_cell(x, y):
        return (int((x - min_x) / cell_size), int((y - min_y) / cell_size))

    def check_collision(x, y, check_dist):
        cell = get_cell(x, y)
        for dx in [-1, 0, 1]:
            for dy in [-1, 0, 1]:
                neighbor = (cell[0] + dx, cell[1] + dy)
                if neighbor in grid:
                    for px, py in grid[neighbor]:
                        dist_sq = (x - px)**2 + (y - py)**2
                        if dist_sq < check_dist**2:
                            return True
        return False

    def add_position(x, y):
        cell = get_cell(x, y)
        if cell not in grid:
            grid[cell] = []
        grid[cell].append((x, y))
        positions.append((x, y))

    def is_in_bounds(x, y, m=None):
        if m is None:
            m = margin
        return (min_x + m <= x <= max_x - m and
                min_y + m <= y <= max_y - m)

    first_x = rng.uniform(-margin * 0.5, margin * 0.5)
    first_y = rng.uniform(-margin * 0.5, margin * 0.5)
    add_position(first_x, first_y)

    active_list = [0]
    fail_counts = {0: 0}
    max_fails_per_point = 8

    for i in range(1, num_objects):
        check_timeout()
        placed = False
        current_min_dist = min_distance
        total_attempts = 0

        while active_list and total_attempts < max_attempts * 0.75 and not placed:
            active_idx = int(rng.randint(0, len(active_list)))
            ref_pos_idx = active_list[active_idx]
            ref_x, ref_y = positions[ref_pos_idx]

            local_attempts = 0
            max_local = 6

            while local_attempts < max_local and not placed:
                angle = float(rng.uniform(0, 2 * np.pi))
                dist = float(rng.uniform(min_distance, min_distance * 1.35))
                x = ref_x + dist * np.cos(angle)
                y = ref_y + dist * np.sin(angle)

                if is_in_bounds(x, y) and not check_collision(x, y, current_min_dist):
                    add_position(x, y)
                    new_idx = len(positions) - 1
                    active_list.append(new_idx)
                    fail_counts[new_idx] = 0
                    placed = True

                local_attempts += 1
                total_attempts += 1

            if not placed:
                fail_counts[ref_pos_idx] = fail_counts.get(ref_pos_idx, 0) + 1
                if fail_counts[ref_pos_idx] >= max_fails_per_point:
                    active_list.pop(active_idx)

        if not placed and active_list:
            current_min_dist = min_distance * 0.8
            attempts_phase2 = 0
            max_phase2 = max_attempts // 4

            while active_list and attempts_phase2 < max_phase2 and not placed:
                active_idx = int(rng.randint(0, len(active_list)))
                ref_pos_idx = active_list[active_idx]
                ref_x, ref_y = positions[ref_pos_idx]

                for _ in range(4):
                    angle = float(rng.uniform(0, 2 * np.pi))
                    dist = float(rng.uniform(current_min_dist, min_distance * 1.3))
                    x = ref_x + dist * np.cos(angle)
                    y = ref_y + dist * np.sin(angle)

                    if is_in_bounds(x, y, margin * 0.5) and not check_collision(x, y, current_min_dist):
                        add_position(x, y)
                        new_idx = len(positions) - 1
                        active_list.append(new_idx)
                        fail_counts[new_idx] = 0
                        placed = True
                        break

                    attempts_phase2 += 1

                if not placed:
                    fail_counts[ref_pos_idx] = fail_counts.get(ref_pos_idx, 0) + 2
                    if fail_counts[ref_pos_idx] >= max_fails_per_point:
                        active_list.pop(active_idx)

        if not placed:
            current_min_dist = min_distance * 0.65
            remaining_attempts = max(10, max_attempts - total_attempts)

            for _ in range(remaining_attempts):
                x = float(rng.uniform(min_x + margin * 0.5, max_x - margin * 0.5))
                y = float(rng.uniform(min_y + margin * 0.5, max_y - margin * 0.5))

                if not check_collision(x, y, current_min_dist):
                    add_position(x, y)
                    new_idx = len(positions) - 1
                    active_list.append(new_idx)
                    fail_counts[new_idx] = 0
                    placed = True
                    break

        if not placed:
            best_x, best_y = None, None
            best_min_dist = -1.0
            for _ in range(20):
                x = float(rng.uniform(min_x + margin * 0.3, max_x - margin * 0.3))
                y = float(rng.uniform(min_y + margin * 0.3, max_y - margin * 0.3))
                if positions:
                    min_dist_to_existing = min(
                        float(np.sqrt((x - px)**2 + (y - py)**2)) for px, py in positions
                    )
                    if min_dist_to_existing > best_min_dist:
                        best_min_dist = min_dist_to_existing
                        best_x, best_y = x, y
            if best_x is not None:
                add_position(best_x, best_y)
            else:
                x = float(rng.uniform(min_x + margin * 0.3, max_x - margin * 0.3))
                y = float(rng.uniform(min_y + margin * 0.3, max_y - margin * 0.3))
                add_position(x, y)

    if len(positions) > 1:
        xs = [p[0] for p in positions]
        ys = [p[1] for p in positions]
        extent_x = max(xs) - min(xs)
        extent_y = max(ys) - min(ys)
        center_x = (min(xs) + max(xs)) / 2
        center_y = (min(ys) + max(ys)) / 2
    else:
        extent_x = extent_y = 1.0
        center_x = center_y = 0.0

    grid_info = {
        "pattern": "compact_random_v2",
        "min_distance": float(min_distance),
        "extent_x": float(extent_x),
        "extent_y": float(extent_y),
        "center": [float(center_x), float(center_y)],
        "actual_count": len(positions),
        "active_list_final_size": len(active_list)
    }
    return positions, grid_info

def place_object_at_position(obj, position):
    x, y = position
    bounds = obj.aabbox
    min_z = bounds[0][2]
    obj.position = (x, y, -min_z + 0.01)

def apply_color_to_object(obj, renderer, color):
    try:
        bl_obj = obj.linked_objects[renderer]
        r, g, b = color

        bl_obj.data.materials.clear()
        mat = bpy.data.materials.new(name=f"ForcedColor_{obj.uid}")
        mat.use_nodes = True
        nodes = mat.node_tree.nodes
        links = mat.node_tree.links
        nodes.clear()

        output = nodes.new('ShaderNodeOutputMaterial')
        output.location = (300, 0)
        principled = nodes.new('ShaderNodeBsdfPrincipled')
        principled.location = (0, 0)
        principled.inputs['Base Color'].default_value = (r, g, b, 1.0)
        principled.inputs['Metallic'].default_value = 0.0
        principled.inputs['Roughness'].default_value = 0.4
        principled.inputs['Specular'].default_value = 0.5
        links.new(principled.outputs['BSDF'], output.inputs['Surface'])
        bl_obj.data.materials.append(mat)
        return True
    except Exception as e:
        logging.debug(f"Failed to apply color: {e}")
        return False

def generate_distinct_sizes(rng, base_size,
                            small_ratio_min, small_ratio_max,
                            large_ratio_min, large_ratio_max):
    small_size = base_size * float(rng.uniform(small_ratio_min, small_ratio_max))
    large_size = base_size * float(rng.uniform(large_ratio_min, large_ratio_max))
    return small_size, large_size

# ============================================================================
# Asset loading helpers
# ============================================================================
def get_shapenet_assets_by_category(shapenet_source):
    category_assets = {}
    for asset_id, spec in shapenet_source._assets.items():
        metadata = spec.get("metadata", {})
        category = metadata.get("category", metadata.get("class", metadata.get("label", "unknown")))
        if category not in category_assets:
            category_assets[category] = []
        category_assets[category].append(asset_id)
    return category_assets

def get_trellis_assets_by_category(trellis_source):
    category_assets = {}
    trellis_asset_ids = set()
    for asset_id, spec in trellis_source._assets.items():
        metadata = spec.get("metadata", {})
        category = metadata.get("category", metadata.get("class", metadata.get("label", None)))
        if category is None:
            category = asset_id[:8]
        if category not in category_assets:
            category_assets[category] = []
        category_assets[category].append(asset_id)
        trellis_asset_ids.add(asset_id)
    return category_assets, trellis_asset_ids

def get_asset_source(asset_id, trellis_asset_set):
    return 'trellis' if asset_id in trellis_asset_set else 'shapenet'

def split_hdri_assets(hdri_source, rng_seed=42):
    all_ids = list(hdri_source._assets.keys())
    rng = np.random.default_rng(rng_seed)
    rng.shuffle(all_ids)
    n = len(all_ids)
    n_test = max(1, int(n / 11))
    test_ids = all_ids[:n_test]
    train_ids = all_ids[n_test:]
    return {'train': train_ids, 'test': test_ids}

# ============================================================================
# Deterministic per-category testA split
# ============================================================================
def stable_str_hash32(s):
    # FNV-1a 32-bit, stable across runs/platforms
    h = 2166136261
    for b in s.encode("utf-8"):
        h ^= b
        h = (h * 16777619) & 0xFFFFFFFF
    return int(h)

def deterministic_testA_split_for_category(asset_ids, fraction, seed, min_keep=2):
    """
    Split asset_ids into (train_ids, testA_ids) deterministically.

    Guarantees:
      - len(train_ids) >= min_keep (if possible)
      - len(testA_ids) >= min_keep (if possible)
    """
    asset_ids = list(asset_ids)
    n = len(asset_ids)
    if n == 0:
        return [], []
    if n <= (2 * min_keep):
        # too few: keep all in train, none in testA
        # (testA would break ">=2" otherwise)
        return asset_ids, []

    rng = np.random.default_rng(seed)
    # stable shuffle: do not depend on incoming order
    ids = sorted(asset_ids)
    rng.shuffle(ids)

    raw = int(round(n * fraction))
    test_sz = max(min_keep, raw)
    test_sz = min(test_sz, n - min_keep)  # keep at least min_keep in train

    test_ids = ids[:test_sz]
    train_ids = ids[test_sz:]
    return train_ids, test_ids

def build_objects_split_assets(category_assets_all, objects_split, fraction, seed, min_keep=2):
    """
    Returns:
      category_assets_filtered: dict category -> list[asset_id] according to split rule
      split_info: dict with stats (for metadata)
    """
    train_categories = set(get_train_categories())
    testB_categories = set(get_testB_categories())

    # filter categories by split
    if objects_split == "testB":
        allowed_categories = testB_categories
    else:
        # train or testA use train categories only
        allowed_categories = train_categories

    # build per-category filtered asset lists
    category_assets_filtered = {}
    split_stats = {
        "objects_split": objects_split,
        "testA_fraction": float(fraction),
        "testA_seed": int(seed),
        "min_keep_per_split": int(min_keep),
        "num_categories": 0,
        "num_assets_total": 0,
        "num_assets_testA_total": 0,
    }

    for cat, ids in category_assets_all.items():
        if cat not in allowed_categories:
            continue

        if objects_split == "testB":
            # no asset-heldout on category OOD split
            kept = list(ids)
            if kept:
                category_assets_filtered[cat] = kept
            continue

        # train/testA: do deterministic split within each category
        train_ids, testA_ids = deterministic_testA_split_for_category(
            ids, fraction=fraction, seed=(seed + stable_str_hash32(cat)), min_keep=min_keep
        )

        if objects_split == "train":
            kept = train_ids
        else:
            kept = testA_ids

        if kept:
            category_assets_filtered[cat] = kept

        split_stats["num_assets_testA_total"] += len(testA_ids)

    split_stats["num_categories"] = len(category_assets_filtered)
    split_stats["num_assets_total"] = sum(len(v) for v in category_assets_filtered.values())
    return category_assets_filtered, split_stats

# ============================================================================
# Weighted sampling utilities
# ============================================================================
def weighted_choice(rng, items, weights):
    w = np.array(weights, dtype=np.float64)
    w = np.clip(w, 0.0, None)
    s = float(w.sum())
    if s <= 0:
        idx = int(rng.randint(0, len(items)))
        return items[idx]
    p = w / s
    idx = int(rng.choice(len(items), p=p))
    return items[idx]

def category_weight(cat, low_thr, low_w):
    c = get_category_count(cat)
    return (low_w if c < low_thr else 1.0)

# ============================================================================
# Count sampling (bugfix)
# ============================================================================
def sample_dual_group_counts(rng, max_total_from_config, total_min=20, total_max=250, group_min=10):
    """
    Sample (c1, c2) with:
      - total in [total_min, total_max] and <= max_total_from_config (but at least 2*group_min)
      - each group >= group_min
    """
    # ensure total upper bound is valid
    hard_min_total = 2 * group_min
    total_min = max(int(total_min), hard_min_total)
    total_max = max(int(total_max), total_min)

    max_total = int(max_total_from_config) if max_total_from_config is not None else total_max
    if max_total < hard_min_total:
        max_total = hard_min_total
    max_total = min(max_total, total_max)

    total = int(rng.randint(total_min, max_total + 1))

    # choose group1 uniformly within feasible range
    g1_min = group_min
    g1_max = total - group_min
    if g1_max < g1_min:
        g1_max = g1_min
    c1 = int(rng.randint(g1_min, g1_max + 1))
    c2 = int(total - c1)
    if c2 < group_min:
        # adjust deterministically
        c2 = group_min
        c1 = total - c2
    return c1, c2

def _sanitize_int(x, default):
    try:
        return int(x)
    except Exception:
        return int(default)

def _shrink_mins_to_fit_cap(min1, min2, cap, rng=None):
    """
    Shrink (min1, min2) so that min1+min2 <= cap, roughly preserving ratio.
    Ensures each min >= 1.
    """
    min1 = max(1, int(min1))
    min2 = max(1, int(min2))
    s = min1 + min2
    if s <= cap:
        return min1, min2

    # proportional shrink
    ratio1 = min1 / float(s)
    new1 = max(1, int(np.floor(cap * ratio1)))
    new2 = max(1, cap - new1)

    # fix if rounding made sum != cap or any got too small
    if new1 + new2 != cap:
        new2 = cap - new1
    if new2 < 1:
        new2 = 1
        new1 = cap - new2
    if new1 < 1:
        new1 = 1
        new2 = cap - new1

    # if still somehow off (extreme), adjust
    while new1 + new2 > cap:
        if new1 > new2 and new1 > 1:
            new1 -= 1
        elif new2 > 1:
            new2 -= 1
        else:
            break

    return new1, new2

def sample_dual_group_counts_v2(rng, min1, min2, max_total, cap_total=250):
    """
    Dual-group count sampling that respects per-group mins from config.

    Rules:
      - Require min1+min2 <= effective max_total.
      - If min1+min2 > max_total: expand effective max_total to min1+min2.
      - If expanded max_total > cap_total: cap at cap_total and shrink mins so sum fits cap_total.
      - Sample total in [min1+min2, effective_max_total], then sample group1 in [min1, total-min2].
    """
    min1 = max(1, _sanitize_int(min1, 1))
    min2 = max(1, _sanitize_int(min2, 1))
    max_total = _sanitize_int(max_total, cap_total)
    cap_total = int(cap_total)

    # Step 1: expand effective max_total if config max_total is too small
    sum_mins = min1 + min2
    effective_max = max(max_total, sum_mins)

    # Step 2: apply hard cap; if cap forces conflict, shrink mins
    if effective_max > cap_total:
        effective_max = cap_total
        if sum_mins > cap_total:
            min1, min2 = _shrink_mins_to_fit_cap(min1, min2, cap_total, rng=rng)
            sum_mins = min1 + min2

    # Step 3: sample total
    if effective_max < sum_mins:
        # should not happen, but guard anyway
        effective_max = sum_mins
    total = int(rng.randint(sum_mins, effective_max + 1))

    # Step 4: sample group1, group2
    g1_low = min1
    g1_high = total - min2
    if g1_high < g1_low:
        # fallback: force boundary
        g1_high = g1_low
    c1 = int(rng.randint(g1_low, g1_high + 1))
    c2 = int(total - c1)

    # final guard
    if c1 < min1:
        c1 = min1
        c2 = total - c1
    if c2 < min2:
        c2 = min2
        c1 = total - c2

    return c1, c2, {"min1_used": int(min1), "min2_used": int(min2), "max_total_used": int(effective_max), "total": int(total)}

def sample_single_group_count(rng, max_total_from_config, total_min=20, total_max=250):
    max_total = int(max_total_from_config) if max_total_from_config is not None else total_max
    if max_total < total_min:
        max_total = total_min
    max_total = min(max_total, total_max)
    return int(rng.randint(total_min, max_total + 1))

def sample_single_group_count_v2(rng, min_count, max_total_from_config, cap_total=250):
    """
    Single-type count sampling that respects config min_objects_per_group.

    Rules:
      - lower bound = min_count
      - upper bound = min(max_total_from_config, cap_total)
      - if min_count > upper bound: set upper bound to cap_total and shrink min_count to cap_total if needed
    """
    min_count = max(1, _sanitize_int(min_count, 1))
    cap_total = int(cap_total)
    max_total = _sanitize_int(max_total_from_config, cap_total)
    max_total = min(max_total, cap_total)

    if max_total < min_count:
        # try to expand up to cap_total
        max_total = cap_total
        if min_count > cap_total:
            min_count = cap_total

    return int(rng.randint(min_count, max_total + 1)), {"min_used": int(min_count), "max_used": int(max_total)}


# ============================================================================
# Preselect logic per Level
# ============================================================================
def preselect_for_level(rng, category_assets, trellis_asset_set):
    """
    Returns dict with fields depending on level setup.
    """
    objects_split = FLAGS.objects_split

    result = {
        "level": FLAGS.level,
        "objects_split": objects_split,
        "category": None,
        "categories": [],
        "super_category": None,
        "asset_id": None,
        "asset_ids": [],
        "asset_id_by_category": {},
        "source_by_asset": {},
    }

    # utility: valid categories with assets
    valid_categories = [c for c, ids in category_assets.items() if len(ids) >= 1]
    if not valid_categories:
        raise RuntimeError(f"No categories available for objects_split={objects_split}")

    # Apply Level4 training filter: exclude count<=50 categories for training only
    if FLAGS.level == 4 and objects_split == "train":
        thr = int(FLAGS.level4_min_count_for_train)
        valid_categories = [c for c in valid_categories if get_category_count(c) >= thr and len(category_assets[c]) >= 2]
        if not valid_categories:
            raise RuntimeError("No categories available for Level4 training after count filter and >=2 assets constraint.")

    if FLAGS.level == 1:
        # pick one category, one asset
        weights = [category_weight(c, FLAGS.low_count_threshold, FLAGS.low_count_weight) for c in valid_categories]
        cat = weighted_choice(rng, valid_categories, weights)
        aid = str(rng.choice(category_assets[cat]))
        result["category"] = cat
        result["asset_id"] = aid
        result["source_by_asset"][aid] = get_asset_source(aid, trellis_asset_set)
        return result

    if FLAGS.level == 2:
        # pick one category, one asset (two groups share asset)
        weights = [category_weight(c, FLAGS.low_count_threshold, FLAGS.low_count_weight) for c in valid_categories]
        cat = weighted_choice(rng, valid_categories, weights)
        aid = str(rng.choice(category_assets[cat]))
        result["category"] = cat
        result["asset_id"] = aid
        result["source_by_asset"][aid] = get_asset_source(aid, trellis_asset_set)
        return result

    if FLAGS.level == 3:
        # two categories within the same super category (like Level5), one asset each
        train_or_testB_key = "train" if objects_split in ["train", "testA"] else "test"

        available_super = []
        super_weights = []

        for sc_name, sc in SUPER_CATEGORIES.items():
            if sc_name == "Structures":
                continue
            cats = sc.get(train_or_testB_key, [])
            cats = [c for c in cats if c in category_assets and len(category_assets[c]) >= 1]
            if len(cats) < 2:
                continue
            available_super.append(sc_name)

            all_low = all(get_category_count(c) < FLAGS.low_count_threshold for c in cats)
            w = (FLAGS.low_count_weight if all_low else 1.0)
            super_weights.append(w)

        if not available_super:
            raise RuntimeError(f"No super categories available for Level3 under objects_split={objects_split}")

        selected_super = weighted_choice(rng, available_super, super_weights)
        cats = SUPER_CATEGORIES[selected_super].get(train_or_testB_key, [])
        cats = [c for c in cats if c in category_assets and len(category_assets[c]) >= 1]

        cat_weights = [category_weight(c, FLAGS.low_count_threshold, FLAGS.low_count_weight) for c in cats]
        cat1 = weighted_choice(rng, cats, cat_weights)
        remaining = [c for c in cats if c != cat1]
        remaining_w = [category_weight(c, FLAGS.low_count_threshold, FLAGS.low_count_weight) for c in remaining]
        cat2 = weighted_choice(rng, remaining, remaining_w)

        aid1 = str(rng.choice(category_assets[cat1]))
        aid2 = str(rng.choice(category_assets[cat2]))

        result["super_category"] = selected_super
        result["categories"] = [cat1, cat2]
        result["asset_id_by_category"] = {cat1: aid1, cat2: aid2}
        result["source_by_asset"][aid1] = get_asset_source(aid1, trellis_asset_set)
        result["source_by_asset"][aid2] = get_asset_source(aid2, trellis_asset_set)
        return result


    if FLAGS.level == 4:
        # same category, two different assets
        candidates = [c for c in valid_categories if len(category_assets[c]) >= 2]
        if not candidates:
            raise RuntimeError("No categories with >=2 assets for Level4.")
        weights = [category_weight(c, FLAGS.low_count_threshold, FLAGS.low_count_weight) for c in candidates]
        cat = weighted_choice(rng, candidates, weights)
        selected_assets = list(rng.choice(category_assets[cat], size=2, replace=False))
        aid1, aid2 = str(selected_assets[0]), str(selected_assets[1])
        result["category"] = cat
        result["asset_ids"] = [aid1, aid2]
        result["source_by_asset"][aid1] = get_asset_source(aid1, trellis_asset_set)
        result["source_by_asset"][aid2] = get_asset_source(aid2, trellis_asset_set)
        return result

    # Level 5: two categories within same super category; multi-asset per category
    # skip "Structures" and require super category has >=2 categories available in this split
    train_or_testB_key = "train" if objects_split in ["train", "testA"] else "test"
    # For testB, super category selection should be among test super categories that have >=2 test categories.
    # But many super categories only define test categories for a subset. We'll only use those with >=2 categories.
    available_super = []
    super_weights = []

    for sc_name, sc in SUPER_CATEGORIES.items():
        if sc_name == "Structures":
            continue
        cats = sc.get(train_or_testB_key, [])
        # need at least 2 categories with available assets
        cats = [c for c in cats if c in category_assets and len(category_assets[c]) >= 1]
        if len(cats) < 2:
            continue

        available_super.append(sc_name)

        # weight rule for super category:
        # if all categories in this super have count < threshold => weight *= low_count_weight
        all_low = all(get_category_count(c) < FLAGS.low_count_threshold for c in cats)
        w = (FLAGS.low_count_weight if all_low else 1.0)
        super_weights.append(w)

    if not available_super:
        raise RuntimeError(f"No super categories available for Level5 under objects_split={objects_split}")

    selected_super = weighted_choice(rng, available_super, super_weights)
    cats = SUPER_CATEGORIES[selected_super].get(train_or_testB_key, [])
    cats = [c for c in cats if c in category_assets and len(category_assets[c]) >= 1]
    # choose 2 distinct categories with category weights
    cat_weights = [category_weight(c, FLAGS.low_count_threshold, FLAGS.low_count_weight) for c in cats]
    cat1 = weighted_choice(rng, cats, cat_weights)
    remaining = [c for c in cats if c != cat1]
    remaining_w = [category_weight(c, FLAGS.low_count_threshold, FLAGS.low_count_weight) for c in remaining]
    cat2 = weighted_choice(rng, remaining, remaining_w)

    result["super_category"] = selected_super
    result["categories"] = [cat1, cat2]
    return result

# ============================================================================
# Level setup functions
# ============================================================================
def setup_level1(rng, sampled_config, preselected):
    cat = preselected["category"]
    aid = preselected["asset_id"]
    source = preselected["source_by_asset"][aid]

    base_size = float(rng.uniform(sampled_config['object_size_min'], sampled_config['object_size_max']))

    group_config = {
        "category": cat,
        "asset_ids": [aid],
        "source": source,
        "size_mode": "fixed",
        "color_mode": "original",
        "target_size": base_size,
        "base_size": base_size,
        "multi_asset": False,
    }
    logging.info(f"  Level 1 - category='{cat}', asset={aid[:20]}..., size={base_size:.3f}, source={source}")
    return [group_config]

def setup_level2(rng, mode, sampled_config, preselected):
    cat = preselected["category"]
    aid = preselected["asset_id"]
    source = preselected["source_by_asset"][aid]

    if mode == "random":
        actual_mode = str(rng.choice(["size", "color"]))
    else:
        actual_mode = mode

    base_size = float(rng.uniform(sampled_config['object_size_min'], sampled_config['object_size_max']))

    if actual_mode == "size":
        small_size, large_size = generate_distinct_sizes(
            rng, base_size,
            sampled_config['level2_small_ratio_min'], sampled_config['level2_small_ratio_max'],
            sampled_config['level2_large_ratio_min'], sampled_config['level2_large_ratio_max'],
        )
        group_configs = [
            {
                "category": cat,
                "asset_ids": [aid],
                "source": source,
                "size_mode": "fixed_small",
                "color_mode": "original",
                "target_size": float(small_size),
                "base_size": base_size,
                "group_label": "small",
                "multi_asset": False,
            },
            {
                "category": cat,
                "asset_ids": [aid],
                "source": source,
                "size_mode": "fixed_large",
                "color_mode": "original",
                "target_size": float(large_size),
                "base_size": base_size,
                "group_label": "large",
                "multi_asset": False,
            },
        ]
        logging.info(f"  Level 2 (size) - category='{cat}', small={small_size:.3f}, large={large_size:.3f}, source={source}")
    else:
        (c1_name, c1_rgb), (c2_name, c2_rgb) = pick_two_distinct_palette_colors(rng, min_dist=0.35)
        group_configs = [
            {
                "category": cat,
                "asset_ids": [aid],
                "source": source,
                "size_mode": "fixed",
                "color_mode": "color1",
                "target_color": c1_rgb,
                "target_color_name": c1_name,
                "target_size": base_size,
                "base_size": base_size,
                "group_label": "color_a",
                "multi_asset": False,
            },
            {
                "category": cat,
                "asset_ids": [aid],
                "source": source,
                "size_mode": "fixed",
                "color_mode": "color2",
                "target_color": c2_rgb,
                "target_color_name": c2_name,
                "target_size": base_size,
                "base_size": base_size,
                "group_label": "color_b",
                "multi_asset": False,
            },
        ]
        logging.info(f"  Level 2 (color) - category='{cat}', color1={c1_name}/{c1_rgb}, color2={c2_name}/{c2_rgb}, source={source}")


    for cfg in group_configs:
        cfg["level2_actual_mode"] = actual_mode

    return group_configs

def setup_level3(rng, sampled_config_cat1, sampled_config_cat2, preselected):
    # two categories, one asset each
    cat1, cat2 = preselected["categories"]
    aid1 = preselected["asset_id_by_category"][cat1]
    aid2 = preselected["asset_id_by_category"][cat2]
    source1 = preselected["source_by_asset"][aid1]
    source2 = preselected["source_by_asset"][aid2]

    base_size1 = float(rng.uniform(sampled_config_cat1['object_size_min'], sampled_config_cat1['object_size_max']))
    base_size2 = float(rng.uniform(sampled_config_cat2['object_size_min'], sampled_config_cat2['object_size_max']))

    group_configs = [
        {
            "category": cat1,
            "asset_ids": [aid1],
            "source": source1,
            "size_mode": "variable",
            "color_mode": "original",
            "target_size": base_size1,
            "base_size": base_size1,
            "multi_asset": False,
            "group_label": "cat_A",
        },
        {
            "category": cat2,
            "asset_ids": [aid2],
            "source": source2,
            "size_mode": "variable",
            "color_mode": "original",
            "target_size": base_size2,
            "base_size": base_size2,
            "multi_asset": False,
            "group_label": "cat_B",
        },
    ]
    logging.info(f"  Level 3 - categoryA='{cat1}', categoryB='{cat2}', sizeA={base_size1:.3f}, sizeB={base_size2:.3f}")
    return group_configs

def setup_level4(rng, sampled_config, preselected):
    # same category, two assets
    cat = preselected["category"]
    aid1, aid2 = preselected["asset_ids"]
    source1 = preselected["source_by_asset"][aid1]
    source2 = preselected["source_by_asset"][aid2]

    base_size = float(rng.uniform(sampled_config['object_size_min'], sampled_config['object_size_max']))

    group_configs = [
        {
            "category": cat,
            "asset_ids": [aid1],
            "source": source1,
            "size_mode": "variable",
            "color_mode": "original",
            "target_size": base_size,
            "base_size": base_size,
            "multi_asset": False,
            "group_label": "asset_A",
        },
        {
            "category": cat,
            "asset_ids": [aid2],
            "source": source2,
            "size_mode": "variable",
            "color_mode": "original",
            "target_size": base_size,
            "base_size": base_size,
            "multi_asset": False,
            "group_label": "asset_B",
        },
    ]
    logging.info(f"  Level 4 - category='{cat}', assetA={aid1[:18]}..., assetB={aid2[:18]}..., size_base={base_size:.3f}")
    return group_configs

def setup_level5(rng, category_assets, sampled_config1, sampled_config2, preselected, trellis_asset_set):
    super_cat = preselected["super_category"]
    cat1, cat2 = preselected["categories"]

    # choose base sizes per category
    size1 = float(rng.uniform(sampled_config1['object_size_min'], sampled_config1['object_size_max']))
    size2 = float(rng.uniform(sampled_config2['object_size_min'], sampled_config2['object_size_max']))

    # enforce max size ratio by scaling up the smaller one (ignoring its config bounds)
    if size1 <= 0 or size2 <= 0:
        size1 = max(size1, 0.1)
        size2 = max(size2, 0.1)
    big = max(size1, size2)
    small = min(size1, size2)
    ratio_before = big / small if small > 1e-6 else 999.0
    ratio_after = ratio_before

    if ratio_before > FLAGS.level5_max_size_ratio:
        small_new = big / float(FLAGS.level5_max_size_ratio)
        if size1 < size2:
            size1 = small_new
        else:
            size2 = small_new
        ratio_after = big / min(size1, size2)

    # pick multi assets per category
    min_assets = int(FLAGS.level5_min_assets_per_category)
    max_assets = int(FLAGS.level5_max_assets_per_category)

    def pick_asset_list(cat):
        all_assets = category_assets[cat]
        actual_max = min(len(all_assets), max_assets)
        actual_min = min(len(all_assets), min_assets)
        if actual_min < 1:
            actual_min = 1
        if actual_max < actual_min:
            actual_max = actual_min
        k = int(rng.randint(actual_min, actual_max + 1))
        return list(rng.choice(all_assets, size=k, replace=False))

    asset_list1 = pick_asset_list(cat1)
    asset_list2 = pick_asset_list(cat2)

    source1 = get_asset_source(asset_list1[0], trellis_asset_set)
    source2 = get_asset_source(asset_list2[0], trellis_asset_set)

    group_configs = [
        {
            "category": cat1,
            "asset_ids": asset_list1,
            "source": source1,
            "size_mode": "fixed",
            "color_mode": "original",
            "target_size": size1,
            "base_size": size1,
            "multi_asset": True,
            "sampled_config": sampled_config1,
        },
        {
            "category": cat2,
            "asset_ids": asset_list2,
            "source": source2,
            "size_mode": "fixed",
            "color_mode": "original",
            "target_size": size2,
            "base_size": size2,
            "multi_asset": True,
            "sampled_config": sampled_config2,
        },
    ]

    logging.info(f"  Level 5 - super='{super_cat}': '{cat1}'({len(asset_list1)} types, size={size1:.3f}) "
                 f"vs '{cat2}'({len(asset_list2)} types, size={size2:.3f}), ratio {ratio_before:.2f}->{ratio_after:.2f}")
    preselected["level5_size_ratio_before"] = float(ratio_before)
    preselected["level5_size_ratio_after"] = float(ratio_after)
    return group_configs

# ============================================================================
# Main program
# ============================================================================
try:
    scene, rng, output_dir, scratch_dir = kb.setup(FLAGS)
    rng = RNG(rng)
    simulator = PyBullet(scene, scratch_dir)
    renderer = Blender(scene, scratch_dir, samples_per_pixel=512)

    # GPU setup
    try:
        bpy.context.preferences.addons['cycles'].preferences.compute_device_type = 'CUDA'
        bpy.context.preferences.addons['cycles'].preferences.get_devices()
        for device in bpy.context.preferences.addons['cycles'].preferences.devices:
            if device.type in ('CUDA', 'OPTIX'):
                device.use = True
        bpy.context.scene.cycles.device = 'GPU'
        logging.info("GPU rendering enabled")
    except Exception as e:
        logging.warning(f"GPU setup failed: {e}")

    kubasic = kb.AssetSource.from_manifest(FLAGS.kubasic_assets)

    # External config
    external_config = None
    if FLAGS.config_file:
        external_config = load_config_from_file(FLAGS.config_file)
        if external_config:
            logging.info(f"Loaded external config from {FLAGS.config_file}")

    # ============================================================================
    # HDRI backgrounds
    # ============================================================================
    hdri_sources = {}
    hdri_splits = {}
    for name, path in [('hdri_haven', FLAGS.hdri_assets), ('hdri_t2l', FLAGS.hdri_t2l_assets)]:
        try:
            hdri_sources[name] = kb.AssetSource.from_manifest(path)
            hdri_splits[name] = split_hdri_assets(hdri_sources[name], rng_seed=42)
            logging.info(f"Loaded {name}: train={len(hdri_splits[name]['train'])}, test={len(hdri_splits[name]['test'])}")
        except Exception as e:
            logging.warning(f"Failed to load {name}: {e}")

    all_hdri = []
    for sn, sp in hdri_splits.items():
        for hid in sp[FLAGS.backgrounds_split]:
            all_hdri.append((sn, hid))
    if not all_hdri:
        raise RuntimeError("No HDRI backgrounds available!")

    hdri_source_name, hdri_id = all_hdri[int(rng.randint(0, len(all_hdri)))]
    background_hdri = hdri_sources[hdri_source_name].create(asset_id=hdri_id)
    scene.metadata["background"] = hdri_id
    scene.metadata["background_source"] = hdri_source_name
    renderer._set_ambient_light_hdri(background_hdri.filename)
    logging.info(f"Background: {hdri_source_name}/{hdri_id} (split={FLAGS.backgrounds_split})")

    # ============================================================================
    # Load ShapeNet + Trellis
    # ============================================================================
    logging.info("Loading ShapeNet assets...")
    check_timeout()

    shapenet_source = kb.AssetSource.from_manifest(FLAGS.shapenet_assets)
    shapenet_by_cat = get_shapenet_assets_by_category(shapenet_source)

    trellis_source = None
    trellis_by_cat = {}
    trellis_asset_set = set()
    if os.path.exists(FLAGS.trellis_assets):
        try:
            logging.info("Loading Trellis assets...")
            trellis_source = kb.AssetSource.from_manifest(FLAGS.trellis_assets)
            trellis_by_cat, trellis_asset_set = get_trellis_assets_by_category(trellis_source)
            logging.info(f"Loaded Trellis: {len(trellis_by_cat)} categories, {sum(len(v) for v in trellis_by_cat.values())} assets")
        except Exception as e:
            logging.warning(f"Failed to load Trellis assets: {e}")
            trellis_source = None
    else:
        logging.warning(f"Trellis assets file not found: {FLAGS.trellis_assets}")

    # merge
    category_assets_all = {}
    for cat, ids in shapenet_by_cat.items():
        category_assets_all.setdefault(cat, []).extend(ids)
    for cat, ids in trellis_by_cat.items():
        category_assets_all.setdefault(cat, []).extend(ids)

    # remove empty
    category_assets_all = {c: list(v) for c, v in category_assets_all.items() if len(v) > 0}

    if not category_assets_all:
        raise RuntimeError("No assets loaded from ShapeNet/Trellis manifests.")

    # ============================================================================
    # Apply objects split filtering (train/testA/testB)
    # ============================================================================
    category_assets, split_stats = build_objects_split_assets(
        category_assets_all,
        objects_split=FLAGS.objects_split,
        fraction=FLAGS.testA_fraction,
        seed=FLAGS.testA_split_seed,
        min_keep=2
    )

    if not category_assets:
        raise RuntimeError(f"No assets available after split filtering: objects_split={FLAGS.objects_split}")

    logging.info(f"Objects split '{FLAGS.objects_split}': {split_stats['num_categories']} categories, "
                 f"{split_stats['num_assets_total']} assets (testA_total={split_stats['num_assets_testA_total']})")

    # ============================================================================
    # Determine mode
    # ============================================================================
    is_single_type = (FLAGS.level == 1)
    if is_single_type:
        logging.info(f"*** LEVEL {FLAGS.level}: SINGLE-TYPE MODE ***")
    else:
        logging.info(f"*** LEVEL {FLAGS.level}: DUAL-TYPE MODE ***")

    # ============================================================================
    # Preselect categories/assets according to level
    # ============================================================================
    logging.info(f"Preselecting for Level {FLAGS.level} ...")
    preselected = preselect_for_level(rng, category_assets, trellis_asset_set)

    # ============================================================================
    # Sample configs (category-wise when needed)
    # ============================================================================
    camera_config = None
    camera_config_source = None

    if FLAGS.level in [1, 2, 4]:
        cat = preselected["category"]
        cfg_list = get_config_for_category(cat, external_config)
        sampled_config = sample_config(cfg_list, rng)
        camera_config = sampled_config
        camera_config_source = cat

    elif FLAGS.level == 3:
        cat1, cat2 = preselected["categories"]
        cfg_list1 = get_config_for_category(cat1, external_config)
        cfg_list2 = get_config_for_category(cat2, external_config)
        sampled_config1 = sample_config(cfg_list1, rng)
        sampled_config2 = sample_config(cfg_list2, rng)
        # camera: choose the one with larger distance_max
        if sampled_config1["camera_distance_max"] >= sampled_config2["camera_distance_max"]:
            camera_config = sampled_config1
            camera_config_source = cat1
        else:
            camera_config = sampled_config2
            camera_config_source = cat2

    else:  # level 5
        cat1, cat2 = preselected["categories"]
        cfg_list1 = get_config_for_category(cat1, external_config)
        cfg_list2 = get_config_for_category(cat2, external_config)
        sampled_config1 = sample_config(cfg_list1, rng)
        sampled_config2 = sample_config(cfg_list2, rng)
        if sampled_config1["camera_distance_max"] >= sampled_config2["camera_distance_max"]:
            camera_config = sampled_config1
            camera_config_source = cat1
        else:
            camera_config = sampled_config2
            camera_config_source = cat2

    logging.info(f"Camera config source: '{camera_config_source}' (dist_max={camera_config['camera_distance_max']:.1f})")

    # ============================================================================
    # Setup groups per level
    # ============================================================================
    if FLAGS.level == 1:
        group_configs = setup_level1(rng, sampled_config, preselected)
    elif FLAGS.level == 2:
        group_configs = setup_level2(rng, FLAGS.level2_mode, sampled_config, preselected)
    elif FLAGS.level == 3:
        cat1, cat2 = preselected["categories"]
        group_configs = setup_level3(rng, sampled_config1, sampled_config2, preselected)
    elif FLAGS.level == 4:
        group_configs = setup_level4(rng, sampled_config, preselected)
    else:
        group_configs = setup_level5(rng, category_assets, sampled_config1, sampled_config2, preselected, trellis_asset_set)

    logging.info("Group configurations:")
    for i, cfg in enumerate(group_configs):
        if cfg.get("multi_asset", False):
            logging.info(f"  Group {i+1}: category='{cfg['category']}', assets={len(cfg['asset_ids'])} types, "
                         f"size={cfg['target_size']:.3f}, size_mode={cfg['size_mode']}")
        else:
            logging.info(f"  Group {i+1}: category='{cfg['category']}', asset={cfg['asset_ids'][0][:20]}..., "
                         f"size={cfg['target_size']:.3f}, size_mode={cfg['size_mode']}, color_mode={cfg['color_mode']}")

    # ============================================================================
    # Sample object counts (bugfix + hard constraints)
    # ============================================================================
    if FLAGS.level == 1:
        min1 = camera_config.get("min_objects_per_group", 1)
        max_total_cfg = camera_config.get("max_total_objects", 250)
        total, single_debug = sample_single_group_count_v2(
            rng,
            min_count=min1,
            max_total_from_config=max_total_cfg,
            cap_total=250
        )
        group_counts = [int(total), 0]
        scene.metadata["count_sampling"] = {**scene.metadata.get("count_sampling", {}), "level1": single_debug}

    else:
        # Determine per-group mins from config (respect config mins)
        if FLAGS.level == 5:
            min1 = sampled_config1.get("min_objects_per_group", 10)
            min2 = sampled_config2.get("min_objects_per_group", 10)
        elif FLAGS.level == 3:
            # Level3 has two categories; use each category's sampled config mins
            min1 = sampled_config1.get("min_objects_per_group", 10)
            min2 = sampled_config2.get("min_objects_per_group", 10)
        else:
            # Level2/Level4 share one config
            min1 = camera_config.get("min_objects_per_group", 10)
            min2 = camera_config.get("min_objects_per_group", 10)

        max_total_cfg = camera_config.get("max_total_objects", 250)

        c1, c2, count_debug = sample_dual_group_counts_v2(
            rng,
            min1=min1,
            min2=min2,
            max_total=max_total_cfg,
            cap_total=250
        )
        group_counts = [c1, c2]
        scene.metadata["count_sampling"] = count_debug


    total_objects = int(sum(group_counts))
    logging.info(f"Object counts: Group1={group_counts[0]}, Group2={group_counts[1]}, Total={total_objects}")

    # ============================================================================
    # Compute avg size and spawn region
    # ============================================================================
    if FLAGS.level == 5:
        size1, size2 = group_configs[0]["target_size"], group_configs[1]["target_size"]
        avg_size = (size1 * group_counts[0] + size2 * group_counts[1]) / float(total_objects)
        density_factor = max(sampled_config1["density_factor"], sampled_config2["density_factor"])
    else:
        avg_size = float(np.mean([cfg["target_size"] for cfg in group_configs]))
        density_factor = float(camera_config["density_factor"])

    SPAWN_REGION, spawn_size = compute_spawn_region(total_objects, avg_size, density_factor)
    logging.info(f"Average object size: {avg_size:.2f}")

    # ============================================================================
    # Create dome
    # ============================================================================
    dome = kubasic.create(asset_id="dome", name="dome", friction=FLAGS.floor_friction,
                          restitution=FLAGS.floor_restitution, static=True, background=True)
    dome.scale = spawn_size / 5.0 + 0.5
    scene += dome
    dome_blender = dome.linked_objects[renderer]
    texture_node = dome_blender.data.materials[0].node_tree.nodes["Image Texture"]
    texture_node.image = bpy.data.images.load(background_hdri.filename)

    # ============================================================================
    # Positions
    # ============================================================================
    if FLAGS.level == 5:
        min_distance_ratio = min(sampled_config1["min_distance_ratio"], sampled_config2["min_distance_ratio"])
        placement_attempts = max(int(sampled_config1["placement_attempts"]), int(sampled_config2["placement_attempts"]))
    else:
        min_distance_ratio = float(camera_config["min_distance_ratio"])
        placement_attempts = int(camera_config["placement_attempts"])

    positions, grid_info = compute_compact_positions(
        total_objects, avg_size, SPAWN_REGION, rng,
        min_distance_ratio=min_distance_ratio,
        max_attempts=placement_attempts
    )
    rng.shuffle(positions)
    group1_positions = positions[:group_counts[0]]
    group2_positions = positions[group_counts[0]:group_counts[0] + group_counts[1]] if group_counts[1] > 0 else []
    logging.info(f"Positions: Group1={len(group1_positions)}, Group2={len(group2_positions)}")

    # ============================================================================
    # Place objects
    # ============================================================================
    placed_objects = []
    meshes_to_remove = []
    template_meshes = {}
    asset_usage_count = {}

    start_time = time.time()
    placement_idx = 0

    all_group_positions = [group1_positions]
    if len(group_configs) > 1:
        all_group_positions.append(group2_positions)

    for group_idx, (cfg, group_positions) in enumerate(zip(group_configs, all_group_positions)):
        if not group_positions:
            continue

        group_label = f"group_{group_idx + 1}"
        is_multi_asset = bool(cfg.get("multi_asset", False))

        # choose source
        if cfg.get("source") == "trellis" and trellis_source is not None:
            current_asset_source = trellis_source
            asset_source_name = "trellis"
        else:
            current_asset_source = shapenet_source
            asset_source_name = "shapenet"

        # per-group config for variable size
        if FLAGS.level in [3, 5]:
            group_sampled_config = cfg.get("sampled_config", camera_config)
        else:
            group_sampled_config = camera_config

        logging.info(f"Placing {group_label}: {len(group_positions)} objects, category='{cfg['category']}', "
                     f"multi_asset={is_multi_asset}, source={asset_source_name}")

        for pos_idx, position in enumerate(group_positions):
            check_timeout()

            if is_multi_asset:
                asset_id = str(rng.choice(cfg["asset_ids"]))
            else:
                asset_id = str(cfg["asset_ids"][0])

            asset_usage_count[asset_id] = asset_usage_count.get(asset_id, 0) + 1

            if cfg["size_mode"] == "variable":
                size_var = float(rng.uniform(group_sampled_config["size_variation_min"],
                                             group_sampled_config["size_variation_max"]))
                target_size = float(cfg["target_size"] * size_var)
            else:
                size_var = 1.0
                target_size = float(cfg["target_size"])

            try:
                obj = current_asset_source.create(asset_id=asset_id)
                assert isinstance(obj, kb.FileBasedObject)

                base_rot = kb.Quaternion(axis=[1, 0, 0], degrees=90)
                z_rot = kb.Quaternion(axis=[0, 0, 1], degrees=float(rng.uniform(0, 360)))
                obj.quaternion = z_rot * base_rot

                scale_factor, original_size = compute_safe_scale(obj, target_size)
                obj.scale = scale_factor

                obj.metadata.update({
                    "group_index": group_idx,
                    "group_label": group_label,
                    "category": cfg["category"],
                    "asset_id": asset_id,
                    "source": asset_source_name,
                    "original_size": float(original_size),
                    "target_size": float(target_size),
                    "final_size": float(original_size * scale_factor),
                    "scale_factor": float(scale_factor),
                    "size_variation": float(size_var),
                    "size_mode": cfg["size_mode"],
                    "color_mode": cfg["color_mode"],
                    "color_applied": False,
                    "applied_color": None,
                    "instance_id": int(placement_idx),
                    "level": int(FLAGS.level),
                    "multi_asset_mode": bool(is_multi_asset),
                    "is_single_type": bool(is_single_type),
                })

                # Level2 color group-level known colors
                if FLAGS.level == 2 and cfg["color_mode"] in ["color1", "color2"] and cfg.get("target_color") is not None:
                    obj.metadata["target_color"] = list(cfg["target_color"])
                    if cfg.get("target_color_name") is not None:
                        obj.metadata["target_color_name"] = str(cfg["target_color_name"])

                scene += obj
                place_object_at_position(obj, position)
                obj.velocity = (0, 0, 0)
                obj.angular_velocity = (0, 0, 0)
                placed_objects.append(obj)

                if cfg["color_mode"] in ["color1", "color2"] and cfg.get("target_color") is not None:
                    ok = apply_color_to_object(obj, renderer, cfg["target_color"])
                    if ok:
                        obj.metadata["color_applied"] = True
                        obj.metadata["applied_color"] = list(cfg["target_color"])
                        obj.metadata["color_group"] = cfg.get("group_label", cfg["color_mode"])

                bl_obj = obj.linked_objects[renderer]
                mesh_key = (group_idx, asset_id)
                if not obj.metadata.get("color_applied", False):
                    if mesh_key not in template_meshes:
                        template_meshes[mesh_key] = bl_obj.data
                    elif bl_obj.data != template_meshes[mesh_key]:
                        old_mesh = bl_obj.data
                        bl_obj.data = template_meshes[mesh_key]
                        if old_mesh and old_mesh.users == 0:
                            meshes_to_remove.append(old_mesh)

                placement_idx += 1

                if placement_idx == FLAGS.placement_speed_check_count:
                    elapsed = time.time() - start_time
                    speed = FLAGS.placement_speed_check_count / elapsed if elapsed > 0 else 0.0
                    logging.info(f"Speed check at {placement_idx} objects: {speed:.3f} obj/s")
                    if speed < FLAGS.placement_speed_threshold:
                        raise SlowPlacementError(
                            f"Placement too slow: {speed:.3f} obj/s < {FLAGS.placement_speed_threshold} obj/s"
                        )

                if placement_idx % 50 == 0:
                    elapsed = time.time() - start_time
                    rate = placement_idx / elapsed if elapsed > 0 else 0.0
                    logging.info(f"Placed {placement_idx}/{total_objects} ({rate:.1f}/s)")

            except TimeoutError:
                raise
            except SlowPlacementError:
                raise
            except Exception as e:
                logging.warning(f"Failed to place object {placement_idx}: {e}")

    for mesh in meshes_to_remove:
        try:
            if mesh.users == 0:
                bpy.data.meshes.remove(mesh)
        except Exception:
            pass

    placement_time = time.time() - start_time
    actual_total = len(placed_objects)
    if actual_total == 0:
        raise RuntimeError("Could not place any objects!")

    logging.info(f"Placed {actual_total} objects in {placement_time:.1f}s")

    # ============================================================================
    # Camera setup (use camera_config)
    # ============================================================================
    logging.info("Setting up camera...")
    check_timeout()

    all_positions = [obj.position for obj in placed_objects]
    xs = [p[0] for p in all_positions]
    ys = [p[1] for p in all_positions]
    zs = [p[2] for p in all_positions]

    center_x = (min(xs) + max(xs)) / 2
    center_y = (min(ys) + max(ys)) / 2
    center_z = float(np.mean(zs))
    extent_x = max(xs) - min(xs) + avg_size
    extent_y = max(ys) - min(ys) + avg_size
    extent = max(extent_x, extent_y, 1.0)

    offset_ratio = float(FLAGS.camera_offset_ratio)
    look_mode = str(rng.choice(['center', 'offset', 'corner', 'edge']))

    if look_mode == 'center':
        look_offset_x = float(rng.uniform(-extent_x * 0.1, extent_x * 0.1))
        look_offset_y = float(rng.uniform(-extent_y * 0.1, extent_y * 0.1))
    elif look_mode == 'offset':
        look_offset_x = float(rng.uniform(-extent_x * offset_ratio * 0.5, extent_x * offset_ratio * 0.5))
        look_offset_y = float(rng.uniform(-extent_y * offset_ratio * 0.5, extent_y * offset_ratio * 0.5))
    elif look_mode == 'corner':
        look_offset_x = float(rng.choice([-1, 1]) * extent_x * rng.uniform(0.15, 0.3))
        look_offset_y = float(rng.choice([-1, 1]) * extent_y * rng.uniform(0.15, 0.3))
    else:
        if float(rng.random()) > 0.5:
            look_offset_x = float(rng.choice([-1, 1]) * extent_x * rng.uniform(0.15, 0.3))
            look_offset_y = float(rng.uniform(-extent_y * 0.15, extent_y * 0.15))
        else:
            look_offset_x = float(rng.uniform(-extent_x * 0.15, extent_x * 0.15))
            look_offset_y = float(rng.choice([-1, 1]) * extent_y * rng.uniform(0.15, 0.3))

    look_x = center_x + look_offset_x
    look_y = center_y + look_offset_y
    look_z = center_z + float(rng.uniform(-0.2, 0.4))

    focal_length = float(rng.uniform(camera_config['focal_length_min'], camera_config['focal_length_max']))
    scene.camera = kb.PerspectiveCamera(focal_length=focal_length, sensor_width=32)

    fov_rad = 2 * np.arctan(16 / focal_length)
    fov_deg = float(np.degrees(fov_rad))
    coverage = float(rng.uniform(camera_config['coverage_min'], camera_config['coverage_max']))

    ideal_dist = (extent / 2) / (np.tan(fov_rad / 2) * coverage)
    distance_multiplier = float(rng.uniform(0.85, 1.0))
    camera_distance = float(ideal_dist * distance_multiplier)

    # if camera_distance > camera_config['camera_distance_max']:
    #     camera_distance = float(camera_config['camera_distance_max'])
    #     actual_coverage = (extent / 2) / (np.tan(fov_rad / 2) * camera_distance)
    #     logging.info(f"  Distance capped at max, actual coverage: {actual_coverage:.2f}")

    if camera_distance > camera_config['camera_distance_max']:
        capped = camera_config['camera_distance_max']
        actual_coverage = (extent / 2) / (np.tan(fov_rad / 2) * capped)
        logging.info(f"  Distance capped at max, actual coverage: {actual_coverage:.2f}")

        # soft fallback: allow larger coverage (more cropping) to avoid extreme out-of-frame
        coverage_cap = 1.15
        if actual_coverage > coverage_cap:
            camera_distance = (extent / 2) / (np.tan(fov_rad / 2) * coverage_cap) * distance_multiplier
            logging.info(f"  Soft recenter: set coverage_cap={coverage_cap:.2f}, recomputed dist={camera_distance:.2f}")
        else:
            camera_distance = capped

    camera_distance = max(camera_distance, float(camera_config['camera_distance_min']))

    camera_elevation = float(rng.uniform(camera_config['camera_angle_min'], camera_config['camera_angle_max']))
    camera_azimuth = float(rng.uniform(0, 360))
    elev_rad = np.radians(camera_elevation)
    azim_rad = np.radians(camera_azimuth)

    cam_x = float(look_x + camera_distance * np.cos(elev_rad) * np.cos(azim_rad))
    cam_y = float(look_y + camera_distance * np.cos(elev_rad) * np.sin(azim_rad))
    cam_z = float(np.clip(camera_distance * np.sin(elev_rad),
                          camera_config['camera_height_min'], 2 * camera_config['camera_height_max']))

    scene.camera.position = (cam_x, cam_y, cam_z)
    scene.camera.look_at((look_x, look_y, look_z))

    logging.info(f"Camera: mode={look_mode}, FOV={fov_deg:.1f}°, dist={camera_distance:.1f}m (source={camera_config_source})")

    camera_params = {
        "focal_length": float(focal_length),
        "fov_degrees": float(fov_deg),
        "distance": float(camera_distance),
        "elevation": float(camera_elevation),
        "azimuth": float(camera_azimuth),
        "position": [float(cam_x), float(cam_y), float(cam_z)],
        "look_at": [float(look_x), float(look_y), float(look_z)],
        "look_mode": look_mode,
        "objects_extent": float(extent),
        "coverage_factor": float(coverage),
        "center": [float(center_x), float(center_y), float(center_z)],
        "config_source_category": str(camera_config_source),
    }

    # ============================================================================
    # Physics simulation
    # ============================================================================
    logging.info("Running physics simulation...")
    check_timeout()
    sim_start = time.time()
    animation, collisions = simulator.run(frame_start=0, frame_end=5)
    sim_time = time.time() - sim_start
    logging.info(f"Simulation completed in {sim_time:.1f}s")

    if FLAGS.save_state:
        renderer.save_state(output_dir / "scene.blend")

    # ============================================================================
    # Render
    # ============================================================================
    logging.info("Rendering...")
    check_timeout()
    render_start = time.time()
    data_stack = renderer.render(return_layers=("rgba", "segmentation"))
    render_time = time.time() - render_start
    logging.info(f"Rendering completed in {render_time:.1f}s")

    # ============================================================================
    # Postprocess
    # ============================================================================
    logging.info("Postprocessing...")
    kb.compute_visibility(data_stack["segmentation"], scene.assets)

    # NEW: override visibility for border fragments (so they will be excluded downstream)
    compute_bbox_and_filter_border_fragments(
        data_stack["segmentation"],
        scene.assets,
        ref_percentile=60.0,
        small_ratio=0.1,
        touch_eps_px=2.0       # 1-pixel tolerance
    )

    visible_foreground_assets = [asset for asset in scene.foreground_assets
                                if np.max(asset.metadata["visibility"]) > 0]
    visible_foreground_assets = sorted(
        visible_foreground_assets,
        key=lambda asset: np.sum(asset.metadata["visibility"]),
        reverse=True
    )

    # Re-adjust segmentation indices based on the filtered asset list.
    data_stack["segmentation"] = kb.adjust_segmentation_idxs(
        data_stack["segmentation"], scene.assets, visible_foreground_assets
    )

    scene.metadata["num_instances"] = len(visible_foreground_assets)

    logging.info("Saving outputs...")
    kb.write_image_dict(data_stack, output_dir)

    kb.post_processing.compute_bboxes(data_stack["segmentation"], visible_foreground_assets)


    # ============================================================================
    # Metadata
    # ============================================================================
    instances_info = kb.get_instance_info(scene, visible_foreground_assets)

    for idx, inst in enumerate(instances_info):
        asset = visible_foreground_assets[idx]
        inst["group_index"] = asset.metadata.get("group_index", 0)
        inst["group_label"] = asset.metadata.get("group_label", "unknown")
        inst["category"] = asset.metadata.get("category", "unknown")
        inst["asset_id"] = asset.metadata.get("asset_id", "unknown")
        inst["source"] = asset.metadata.get("source", "shapenet")
        inst["instance_id"] = asset.metadata.get("instance_id", None)
        inst["size_variation"] = asset.metadata.get("size_variation", 1.0)
        inst["target_size"] = asset.metadata.get("target_size", None)
        inst["final_size"] = asset.metadata.get("final_size", None)
        inst["size_mode"] = asset.metadata.get("size_mode", "fixed")
        inst["color_mode"] = asset.metadata.get("color_mode", "original")
        inst["color_applied"] = asset.metadata.get("color_applied", False)
        inst["applied_color"] = asset.metadata.get("applied_color", None)
        inst["multi_asset_mode"] = asset.metadata.get("multi_asset_mode", False)
        inst["is_single_type"] = asset.metadata.get("is_single_type", False)
        if "target_color" in asset.metadata:
            inst["target_color"] = asset.metadata["target_color"]
        if "target_color_name" in asset.metadata:
            inst["target_color_name"] = asset.metadata["target_color_name"]


    # group summary
    visible_counts = [0, 0]
    for asset in visible_foreground_assets:
        gi = int(asset.metadata.get("group_index", 0))
        if gi < 2:
            visible_counts[gi] += 1

    group_info = []
    for i, cfg in enumerate(group_configs):
        info = {
            "group_index": i,
            "group_label": f"group_{i+1}",
            "category": cfg["category"],
            "size_mode": cfg["size_mode"],
            "color_mode": cfg["color_mode"],
            "target_count": int(group_counts[i]) if i < len(group_counts) else 0,
            "visible_count": int(visible_counts[i]) if i < len(visible_counts) else 0,
            "multi_asset": bool(cfg.get("multi_asset", False)),
            "source": cfg.get("source", "shapenet"),
            "target_size": float(cfg["target_size"]),
        }
        if cfg.get("multi_asset", False):
            info["asset_ids"] = list(cfg["asset_ids"])
            info["num_asset_types"] = int(len(cfg["asset_ids"]))
        else:
            info["asset_id"] = str(cfg["asset_ids"][0])

        if cfg.get("target_color") is not None:
            info["target_color"] = list(cfg["target_color"])
        
        if cfg.get("target_color_name") is not None:
            info["target_color_name"] = str(cfg["target_color_name"])

        if cfg.get("level2_actual_mode"):
            info["level2_mode"] = str(cfg["level2_actual_mode"])

        group_info.append(info)

    level_specific_info = {"level": int(FLAGS.level)}
    if FLAGS.level == 1:
        level_specific_info.update({
            "mode": "single_type",
            "category": group_configs[0]["category"],
            "asset_id": group_configs[0]["asset_ids"][0],
        })
    elif FLAGS.level == 2:
        actual_mode = group_configs[0].get("level2_actual_mode", FLAGS.level2_mode)
        level_specific_info.update({
            "mode": str(actual_mode),
            "distinction_type": "size" if actual_mode == "size" else "color",
        })
        if actual_mode == "color":
            level_specific_info["group1_color"] = list(group_configs[0].get("target_color"))
            level_specific_info["group2_color"] = list(group_configs[1].get("target_color"))
            level_specific_info["group1_color_name"] = str(group_configs[0].get("target_color_name"))
            level_specific_info["group2_color_name"] = str(group_configs[1].get("target_color_name"))

    elif FLAGS.level == 3:
        level_specific_info.update({
            "category_1": group_configs[0]["category"],
            "asset_1": group_configs[0]["asset_ids"][0],
            "category_2": group_configs[1]["category"],
            "asset_2": group_configs[1]["asset_ids"][0],
            "super_category": preselected.get("super_category"),
        })
    elif FLAGS.level == 4:
        level_specific_info.update({
            "category": group_configs[0]["category"],
            "asset_1": group_configs[0]["asset_ids"][0],
            "asset_2": group_configs[1]["asset_ids"][0],
        })
    else:
        level_specific_info.update({
            "super_category": preselected.get("super_category"),
            "category_1": group_configs[0]["category"],
            "category_2": group_configs[1]["category"],
            "size_ratio_before": float(preselected.get("level5_size_ratio_before", 0.0)),
            "size_ratio_after": float(preselected.get("level5_size_ratio_after", 0.0)),
            "camera_config_source": str(camera_config_source),
        })

    total_time = time.time() - SCRIPT_START_TIME

    metadata_output = {
        "flags": vars(FLAGS),
        "sampled_config": to_serializable(camera_config),
        "metadata": kb.get_scene_metadata(scene),
        "camera": {**kb.get_camera_info(scene.camera), "random_params": camera_params},
        "instances": instances_info,

        "level": int(FLAGS.level),
        "is_single_type": bool(is_single_type),
        "level_description": {
            1: "Single category, single asset (single-type)",
            2: "Same asset, two different sizes or colors (dual-type)",
            3: "Two categories, one asset each, variable sizes (dual-type)",
            4: "Same category, two different assets, variable sizes (dual-type)",
            5: "Two categories (same super category), multi-asset per category (dual-type)",
        }[int(FLAGS.level)],

        "groups": group_info,
        "level_specific_info": level_specific_info,

        "counting_info": {
            "total_target": int(total_objects),
            "total_actual": int(actual_total),
            "total_visible": int(len(visible_foreground_assets)),
            "group1_target": int(group_counts[0]),
            "group1_visible": int(visible_counts[0]),
            "group2_target": int(group_counts[1]) if len(group_counts) > 1 else 0,
            "group2_visible": int(visible_counts[1]) if len(visible_counts) > 1 else 0,
            "is_single_type": bool(is_single_type),
        },

        "placement_info": {
            "mode": "compact_random_v2",
            "grid_info": grid_info,
            "spawn_region_size": float(spawn_size),
            "objects_extent": float(extent),
        },

        "background_info": {
            "source": hdri_source_name,
            "asset_id": hdri_id,
            "split": FLAGS.backgrounds_split,
        },

        "split_info": {
            "objects_split": FLAGS.objects_split,
            "backgrounds_split": FLAGS.backgrounds_split,
            "testA_fraction": float(FLAGS.testA_fraction),
            "testA_seed": int(FLAGS.testA_split_seed),
            "split_stats": split_stats,
            "trellis_enabled": bool(trellis_source is not None),
        },

        "weighting_info": {
            "low_count_threshold": int(FLAGS.low_count_threshold),
            "low_count_weight": float(FLAGS.low_count_weight),
        },

        "performance": {
            "placement_time": float(placement_time),
            "simulation_time": float(sim_time),
            "render_time": float(render_time),
            "total_time": float(total_time),
            "objects_per_second": float(actual_total / max(placement_time, 0.1)),
        },

        "image_resolution": int(resolution),
        "spawn_region": SPAWN_REGION,
    }

    kb.write_json(filename=output_dir / "metadata.json", data=to_serializable(metadata_output))

    logging.info("=" * 60)
    logging.info(f"Done! Level {FLAGS.level} | objects_split={FLAGS.objects_split} | visible={len(visible_foreground_assets)}/{actual_total}")
    logging.info(f"Time: place={placement_time:.1f}s, sim={sim_time:.1f}s, render={render_time:.1f}s, total={total_time:.1f}s")
    logging.info("=" * 60)

    kb.done()

except SlowPlacementError as e:
    logging.error("=" * 60)
    logging.error(f"SLOW PLACEMENT: {e}")
    logging.error("Terminating to try a different asset...")
    logging.error("=" * 60)
    sys.exit(125)

except TimeoutError as e:
    logging.error("=" * 60)
    logging.error(f"TIMEOUT: {e}")
    logging.error("=" * 60)
    sys.exit(124)

except Exception as e:
    logging.error(f"Error: {e}")
    import traceback
    traceback.print_exc()
    sys.exit(1)
