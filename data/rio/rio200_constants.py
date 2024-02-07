
### ScanNet200 Benchmark constants ###
VALID_CLASS_IDS_200 = (503, 85, 59, 338, 391, 188, 82, 342, 520, 155, 455, 236, 263, 129, 298, 327, 68, 15, 100, 267, 4, 158, 251, 415, 476, 442, 482, 228, 47, 138, 289, 110, 508, 525, 24, 37, 423, 131, 40, 489, 295, 48, 360, 108, 65, 25, 469, 252, 103, 288, 434, 14, 359, 137, 321, 120, 402, 341, 201, 448, 250, 118, 490, 512, 367, 304, 101, 286, 357, 500, 410, 337, 403, 56, 264, 378, 452, 45, 142, 31, 471, 505, 523, 461, 519, 127, 139, 119, 345, 60, 433, 369, 404, 58, 340, 405, 243, 355, 52, 303, 73, 41, 144, 470, 96, 353, 207, 302, 192, 306, 237, 260, 265, 161, 1, 78, 421, 39, 388, 456, 450, 54, 35, 499, 140, 426, 377, 19, 46, 488, 134, 479, 193, 106, 53, 409, 244, 424, 49, 319, 97, 382, 504, 125, 354, 208, 445, 316, 395, 235, 511, 299, 224, 184, 109, 262, 366, 239, 273, 292, 240, 218, 394, 194, 174, 18, 199, 107, 300, 20, 214, 206, 55, 474, 189, 494, 432, 301, 317, 484, 169, 380, 310, 330, 226, 231, 182, 17, 269, 111, 57, 147, 84, 104, 370, 472, 205, 401, 190, 162)
CLASS_LABELS_200 = ('wall', 'chair', 'box', 'pillow', 'shelf', 'floor', 'ceiling', 'plant', 'window', 'door', 'table', 'item', 'lamp', 'curtain', 'object', 'picture', 'cabinet', 'bag', 'clothes', 'light', 'armchair', 'doorframe', 'kitchen cabinet', 'sink', 'towel', 'stool', 'trash can', 'heater', 'blanket', 'desk', 'monitor', 'commode', 'wardrobe', 'windowsill', 'basket', 'bed', 'sofa', 'cushion', 'bench', 'tv', 'nightstand', 'blinds', 'radiator', 'coffee table', 'bucket', 'bath cabinet', 'toilet', 'kitchen counter', 'clutter', 'mirror', 'stand', 'backpack', 'rack', 'decoration', 'pc', 'counter', 'shoes', 'plank', 'frame', 'stove', 'kitchen appliance', 'couch', 'tv stand', 'washing machine', 'refrigerator', 'oven', 'clothes dryer', 'microwave', 'puf', 'vase', 'side table', 'pillar', 'showcase', 'bottle', 'laptop', 'sack', 'suitcase', 'bin', 'dining chair', 'bathtub', 'toilet paper', 'wall frame', 'window frame', 'telephone', 'whiteboard', 'cupboard', 'desk chair', 'couch table', 'plate', 'boxes', 'stairs', 'roll', 'shower', 'bowl', 'pipe', 'shower curtain', 'kettle', 'printer', 'book', 'ottoman', 'candle', 'beverage crate', 'dining table', 'toilet brush', 'clock', 'pot', 'garbage', 'organizer', 'flower', 'pack', 'items', 'kitchen towel', 'laundry basket', 'drawer', 'air conditioner', 'carpet', 'soap dispenser', 'bedside table', 'shades', 'table lamp', 'stuffed animal', 'bookshelf', 'beanbag', 'vacuum cleaner', 'device', 'speaker', 'rug', 'bar', 'blackboard', 'tube', 'cutting board', 'toy', 'flowers', 'coffee machine', 'books', 'shower wall', 'keyboard', 'sofa chair', 'board', 'partition', 'closet', 'scale', 'wall /other room', 'cube', 'price tag', 'garbage bin', 'storage box', 'paper towel', 'shelves', 'ironing board', 'washing basket', 'objects', 'hanger', 'fireplace', 'column', 'ladder', 'recycle bin', 'jalousie', 'luggage', 'napkins', 'jar', 'hand dryer', 'shelf unit', 'flush', 'exhaust hood', 'ball', 'foosball table', 'coffee maker', 'office chair', 'bar stool', 'guitar', 'furniture', 'boots', 'toiletry', 'floor /other room', 'umbrella', 'stair', 'office table', 'paper towel dispenser', 'tray', 'drying machine', 'salt', 'pan', 'pile of books', 'hanging cabinet', 'humidifier', 'file cabinet', 'balcony door', 'locker', 'computer', 'bottles', 'dish dryer', 'ceiling light', 'coat', 'rolled carpet', 'toilet paper dispenser', 'fruits', 'shoe shelf', 'floor lamp', 'drawers')

SCANNET_COLOR_MAP_200 = {
    503: (0.0, 0.0, 0.0),
    85: (174.0, 199.0, 232.0),
    59: (188.0, 189.0, 34.0),
    338: (152.0, 223.0, 138.0),
    391: (255.0, 152.0, 150.0),
    188: (214.0, 39.0, 40.0),
    82: (91.0, 135.0, 229.0),
    342: (31.0, 119.0, 180.0),
    520: (229.0, 91.0, 104.0),
    155: (247.0, 182.0, 210.0),
    455: (91.0, 229.0, 110.0),
    236: (255.0, 187.0, 120.0),
    263: (141.0, 91.0, 229.0),
    129: (112.0, 128.0, 144.0),
    298: (196.0, 156.0, 148.0),
    327: (197.0, 176.0, 213.0),
    68: (44.0, 160.0, 44.0),
    15: (148.0, 103.0, 189.0),
    100: (229.0, 91.0, 223.0),
    267: (219.0, 219.0, 141.0),
    4: (192.0, 229.0, 91.0),
    158: (88.0, 218.0, 137.0),
    251: (58.0, 98.0, 137.0),
    415: (177.0, 82.0, 239.0),
    476: (255.0, 127.0, 14.0),
    442: (237.0, 204.0, 37.0),
    482: (41.0, 206.0, 32.0),
    228: (62.0, 143.0, 148.0),
    47: (34.0, 14.0, 130.0),
    138: (143.0, 45.0, 115.0),
    289: (137.0, 63.0, 14.0),
    110: (23.0, 190.0, 207.0),
    508: (16.0, 212.0, 139.0),
    525: (90.0, 119.0, 201.0),
    24: (125.0, 30.0, 141.0),
    37: (150.0, 53.0, 56.0),
    423: (186.0, 197.0, 62.0),
    131: (227.0, 119.0, 194.0),
    40: (38.0, 100.0, 128.0),
    489: (120.0, 31.0, 243.0),
    295: (154.0, 59.0, 103.0),
    48: (169.0, 137.0, 78.0),
    360: (143.0, 245.0, 111.0),
    108: (37.0, 230.0, 205.0),
    65: (14.0, 16.0, 155.0),
    25: (196.0, 51.0, 182.0),
    469: (237.0, 80.0, 38.0),
    252: (138.0, 175.0, 62.0),
    103: (158.0, 218.0, 229.0),
    288: (38.0, 96.0, 167.0),
    434: (190.0, 77.0, 246.0),
    14: (208.0, 49.0, 84.0),
    359: (208.0, 193.0, 72.0),
    137: (55.0, 220.0, 57.0),
    321: (10.0, 125.0, 140.0),
    120: (76.0, 38.0, 202.0),
    402: (191.0, 28.0, 135.0),
    341: (211.0, 120.0, 42.0),
    201: (118.0, 174.0, 76.0),
    448: (17.0, 242.0, 171.0),
    250: (20.0, 65.0, 247.0),
    118: (208.0, 61.0, 222.0),
    490: (162.0, 62.0, 60.0),
    512: (210.0, 235.0, 62.0),
    367: (45.0, 152.0, 72.0),
    304: (35.0, 107.0, 149.0),
    101: (160.0, 89.0, 237.0),
    286: (227.0, 56.0, 125.0),
    357: (169.0, 143.0, 81.0),
    500: (42.0, 143.0, 20.0),
    410: (25.0, 160.0, 151.0),
    337: (82.0, 75.0, 227.0),
    403: (253.0, 59.0, 222.0),
    56: (240.0, 130.0, 89.0),
    264: (123.0, 172.0, 47.0),
    378: (71.0, 194.0, 133.0),
    452: (24.0, 94.0, 205.0),
    45: (134.0, 16.0, 179.0),
    142: (159.0, 32.0, 52.0),
    31: (213.0, 208.0, 88.0),
    471: (64.0, 158.0, 70.0),
    505: (18.0, 163.0, 194.0),
    523: (65.0, 29.0, 153.0),
    461: (177.0, 10.0, 109.0),
    519: (152.0, 83.0, 7.0),
    127: (83.0, 175.0, 30.0),
    139: (18.0, 199.0, 153.0),
    119: (61.0, 81.0, 208.0),
    345: (213.0, 85.0, 216.0),
    60: (170.0, 53.0, 42.0),
    433: (161.0, 192.0, 38.0),
    369: (23.0, 241.0, 91.0),
    404: (12.0, 103.0, 170.0),
    58: (151.0, 41.0, 245.0),
    340: (133.0, 51.0, 80.0),
    405: (184.0, 162.0, 91.0),
    243: (50.0, 138.0, 38.0),
    355: (31.0, 237.0, 236.0),
    52: (39.0, 19.0, 208.0),
    303: (223.0, 27.0, 180.0),
    73: (254.0, 141.0, 85.0),
    41: (97.0, 144.0, 39.0),
    144: (106.0, 231.0, 176.0),
    470: (12.0, 61.0, 162.0),
    96: (124.0, 66.0, 140.0),
    353: (137.0, 66.0, 73.0),
    207: (250.0, 253.0, 26.0),
    302: (55.0, 191.0, 73.0),
    192: (60.0, 126.0, 146.0),
    306: (153.0, 108.0, 234.0),
    237: (184.0, 58.0, 125.0),
    260: (135.0, 84.0, 14.0),
    265: (139.0, 248.0, 91.0),
    161: (53.0, 200.0, 172.0),
    1: (63.0, 69.0, 134.0),
    78: (190.0, 75.0, 186.0),
    421: (127.0, 63.0, 52.0),
    39: (141.0, 182.0, 25.0),
    388: (56.0, 144.0, 89.0),
    456: (64.0, 160.0, 250.0),
    450: (182.0, 86.0, 245.0),
    54: (139.0, 18.0, 53.0),
    35: (134.0, 120.0, 54.0),
    499: (49.0, 165.0, 42.0),
    140: (51.0, 128.0, 133.0),
    426: (44.0, 21.0, 163.0),
    377: (232.0, 93.0, 193.0),
    19: (176.0, 102.0, 54.0),
    46: (116.0, 217.0, 17.0),
    488: (54.0, 209.0, 150.0),
    134: (60.0, 99.0, 204.0),
    479: (129.0, 43.0, 144.0),
    193: (252.0, 100.0, 106.0),
    106: (187.0, 196.0, 73.0),
    53: (13.0, 158.0, 40.0),
    409: (52.0, 122.0, 152.0),
    244: (128.0, 76.0, 202.0),
    424: (187.0, 50.0, 115.0),
    49: (180.0, 141.0, 71.0),
    319: (77.0, 208.0, 35.0),
    97: (72.0, 183.0, 168.0),
    382: (97.0, 99.0, 203.0),
    504: (172.0, 22.0, 158.0),
    125: (155.0, 64.0, 40.0),
    354: (118.0, 159.0, 30.0),
    208: (69.0, 252.0, 148.0),
    445: (45.0, 103.0, 173.0),
    316: (111.0, 38.0, 149.0),
    395: (184.0, 9.0, 49.0),
    235: (188.0, 174.0, 67.0),
    511: (53.0, 206.0, 53.0),
    299: (97.0, 235.0, 252.0),
    224: (66.0, 32.0, 182.0),
    184: (236.0, 114.0, 195.0),
    109: (241.0, 154.0, 83.0),
    262: (133.0, 240.0, 52.0),
    366: (16.0, 205.0, 144.0),
    239: (75.0, 101.0, 198.0),
    273: (237.0, 95.0, 251.0),
    292: (191.0, 52.0, 49.0),
    240: (227.0, 254.0, 54.0),
    218: (49.0, 206.0, 87.0),
    394: (48.0, 113.0, 150.0),
    194: (125.0, 73.0, 182.0),
    174: (229.0, 32.0, 114.0),
    18: (158.0, 119.0, 28.0),
    199: (60.0, 205.0, 27.0),
    107: (18.0, 215.0, 201.0),
    300: (79.0, 76.0, 153.0),
    20: (134.0, 13.0, 116.0),
    214: (192.0, 97.0, 63.0),
    206: (108.0, 163.0, 18.0),
    55: (95.0, 220.0, 156.0),
    474: (98.0, 141.0, 208.0),
    189: (144.0, 19.0, 193.0),
    494: (166.0, 36.0, 57.0),
    432: (212.0, 202.0, 34.0),
    301: (23.0, 206.0, 34.0),
    317: (91.0, 211.0, 236.0),
    484: (79.0, 55.0, 137.0),
    169: (182.0, 19.0, 117.0),
    380: (134.0, 76.0, 14.0),
    310: (87.0, 185.0, 28.0),
    330: (82.0, 224.0, 187.0),
    226: (92.0, 110.0, 214.0),
    231: (168.0, 80.0, 171.0),
    182: (197.0, 63.0, 51.0),
    17: (175.0, 199.0, 77.0),
    269: (62.0, 180.0, 98.0),
    111: (8.0, 91.0, 150.0),
    57: (77.0, 15.0, 130.0),
    147: (154.0, 65.0, 96.0),
    84: (197.0, 152.0, 11.0),
    104: (59.0, 155.0, 45.0),
    370: (12.0, 147.0, 145.0),
    472: (54.0, 35.0, 219.0),
    205: (210.0, 73.0, 181.0),
    401: (221.0, 124.0, 77.0),
    190: (149.0, 214.0, 66.0),
    162: (72.0, 185.0, 134.0)
}

### For instance segmentation the non-object categories ###
VALID_PANOPTIC_IDS = (188, 189, 82, 83, 503, 504)

# surface_objects = ['floor', 'floor /other room', 'ceiling', 'ceiling /other room', 'wall', 'wall /other room']
CLASS_LABELS_PANOPTIC = ('floor', 'floor /other room', 'ceiling', 'ceiling /other room', 'wall', 'wall /other room')