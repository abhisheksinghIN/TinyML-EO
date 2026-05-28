import numpy as np
from matplotlib import colors

# ============================================================
#                  IMAGE NORMALIZATION (15 channels)
# ============================================================

IMAGE_MEANS = np.array([
    -12.27225428,
    -19.43526044,
    1234.8107257,
    997.61939167,
    903.69102444,
    754.84894802,
    969.18097088,
    1627.91563458,
    1913.47085481,
    1873.95219743,
    2086.10866809,
    710.37421625,
    11.20448001,
    1419.03241856,
    879.26727913
], dtype=np.float32)

IMAGE_STDS = np.array([
    5.18289882,
    6.61391834,
    194.32151121,
    260.6317736,
    300.12094349,
    456.76902268,
    475.76986072,
    876.19360379,
    1101.46597749,
    1122.17195714,
    1230.42499543,
    510.8142223,
    6.90812234,
    948.92739869,
    720.08974123
], dtype=np.float32)


# ==================== Class Mapping (HR GT) ==================== #
#ORIGINAL_TO_REMAPPED_GT = {
#    0: 0,          # Ignore
#    1: 1,        # Forest
#    2: 0,     # Shrubland
#    3: 3,     # Savanna
#    4: 4,      # Grassland
#    5: 5,     # Wetlands
#    6: 6,      # Croplands
#    7: 7,    # Urban
#    8: 8,    # Snow/Ice
#    9: 9,      # Barren
#    10: 10,   # Water
#}
#kept = [1, 4, 6, 7, 10]
ORIGINAL_TO_REMAPPED_GT = {
    0: 0,          # Ignore
    1: 1,        # Forest
    2: 1,     # Shrubland
    3: 0,     # Savanna
    4: 1,      # Grassland
    5: 2,     # Wetlands
    6: 1,      # Croplands
    7: 3,    # Urban
    8: 0,    # Snow/Ice
    9: 0,      # Barren
    10: 4,   # Water
}

def get_label_class_to_idx_map_GT():
    label_to_idx_map = np.zeros(256, dtype=np.int64)
    for raw_label, new_class in ORIGINAL_TO_REMAPPED_GT.items():
        label_to_idx_map[raw_label] = new_class
    return label_to_idx_map

LABEL_CLASS_TO_IDX_MAP_GT = get_label_class_to_idx_map_GT()


# ============================================================
#                    COLORMAP (DFC10)
# ============================================================

#LABEL_CLASS_COLORMAP = {
#    0: (0, 0, 0),          # Ignore
#    1: (0, 153, 0),        # Forest
#    2: (198, 176, 68),     # Shrubland
#    3: (251, 255, 19),     # Savanna
#    4: (182, 255, 5),      # Grassland
#    5: (39, 255, 135),     # Wetlands
#    6: (194, 79, 68),      # Croplands
#    7: (165, 165, 165),    # Urban
#    8: (249, 255, 164),    # Snow/Ice
#    9: (150, 75, 0),        # Barren
#    10: (0, 0, 255),   # Water
#}
LABEL_CLASS_COLORMAP = {
    0: (0, 0, 0),          # Ignore
    1: (0, 180, 140),        # Vegetation
    2: (100, 150, 120),     # Wetland
    3: (200, 0, 120),     # Urban
    4: (40, 120, 200)      # Water
}


LABEL_IDX_COLORMAP = {k: v for k, v in LABEL_CLASS_COLORMAP.items()}

# ============================================================
#                    LABEL NAMES
# ============================================================
LABEL_NAMES = {
    0: "Ignore",
    1: "Vegetation",
    2: "Wetland",
    3: "Urban",
    4: "Water"
}

# ============================================================
#          RGB VISUALIZATION HELPERS
# ============================================================
def class_map_to_rgb(class_map):
    """Convert (H, W) class map to an RGB visualization."""
    rgb = np.zeros((*class_map.shape, 3), dtype=np.uint8)
    for cls_id, color in LABEL_CLASS_COLORMAP.items():
        rgb[class_map == cls_id] = color
    return rgb


