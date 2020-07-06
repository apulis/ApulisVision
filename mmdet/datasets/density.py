import numpy as np
from scipy import ndimage

_categories = (-1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 1, 1, 1, 1, 1, 1, 1, 1, 2, 2, 2, 2, 2, 2, 2, 2, 2,
               3, 3, 3, 3, 3, 3, 3, 3, 3, 3, 3, 4, 4, 4, 4, 4, 4, 4, 4, 4, 4, 4, 4, 5, 5, 5, 5, 5, 5, 5, 5, 5, 5, 5, 5, 5, 5, 5, 5, 5,
               6, 6, 6, 6, 6, 6, 6, 6,
               7, 7, 6, 6, 6, 6, 6, 6, 6, 7, 7, 7, 7, 7, 7, 7, 7, 7, 8, 8, 8, 8, 8, 8, 8, 8, 8, 8, 8, 9, 9, 9, 9, 9, 9, 9, 9, 9, 9, 9, 9, 9, 9,
               10, 10, 10, 10, 10, 10, 10, 10, 10, 10, 10, 10, 11, 11, 11, 11, 11, 11, 11, 11, 12, 12, 12, 12, 12, 12, 12, 12, 12, 12,
               13, 13, 13, 13, 13, 13, 13, 13, 13, 13, 13, 13, 14, 14, 14, 14, 14, 14, 14, 14, 14, 14, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15,
               15, 15, 15, 15, 15, 15, 15, 15, 15,
               16, 16, 16, 16, 16, 16, 16)

RPC_SUPPORT_CATEGORIES = (1, 17, 200)

_coco_categories = (
    -1, 0, 1, 1, 1, 1, 1, 1, 1, 1, 2, 2, 2, 2, 2, 3, 3, 3, 3, 3, 3, 3, 3, 3, 3, 4, 4, 4, 4, 4, 5, 5, 5, 5, 5, 5, 5, 5, 5, 5, 6, 6, 6, 6, 6, 6, 6, 7,
    7,
    7, 7, 7, 7, 7, 7, 7, 7, 8, 8, 8, 8, 8, 8, 9, 9, 9, 9, 9, 9, 10, 10, 10, 10, 10, 11, 11, 11, 11, 11, 11, 11)
COCO_SUPPORT_CATEGORIES = (1, 12, 80)


def contiguous_coco_category_to_super_category(category_id, num_classes):
    cat_id = -1
    assert num_classes in COCO_SUPPORT_CATEGORIES, 'Not support {} density categories'.format(num_classes)
    if num_classes == 12:
        cat_id = _coco_categories[category_id]
    elif num_classes == 1:
        cat_id = 0
    elif num_classes == 80:
        cat_id = category_id - 1
    assert 79 >= cat_id >= 0
    return cat_id


def rpc_category_to_super_category(category_id, num_classes):
    """Map category to super-category id
    Args:
        category_id: list of category ids, 1-based
        num_classes: 1, 17, 200
    Returns:
        super-category id, 0-based
    """
    cat_id = -1
    assert num_classes in RPC_SUPPORT_CATEGORIES, 'Not support {} density categories'.format(num_classes)
    if num_classes == 17:
        cat_id = _categories[category_id]
    elif num_classes == 1:
        cat_id = 0
    elif num_classes == 200:
        cat_id = category_id - 1
    assert 199 >= cat_id >= 0
    return cat_id


def generate_density_map(labels, boxes, scale=50.0 / 800, size=50, num_classes=200, min_sigma=1):
    density_map = np.zeros((num_classes, size, size), dtype=np.float32)
    for category, box in zip(labels, boxes):
        x1, y1, x2, y2 = [x * scale for x in box]
        w, h = x2 - x1, y2 - y1
        box_radius = min(w, h) / 2
        sigma = max(min_sigma, box_radius * 5 / (4 * 3))  # 3/5 of gaussian kernel is in box
        cx, cy = round((x1 + x2) / 2), round((y1 + y2) / 2)
        density = np.zeros((size, size), dtype=np.float32)
        density[cy, cx] = 100
        density = ndimage.filters.gaussian_filter(density, sigma, mode='constant')
        density_map[category, :, :] += density

    return density_map


def generate_density_map_v1(labels, boxes, scale=50.0 / 800, size=50, num_classes=200, min_sigma=1):
    
    num_classes = 3
    density_map = np.zeros((num_classes, size, size), dtype=np.float32)
    for category, box in zip(labels, boxes):
        x1, y1, x2, y2 = [x * scale for x in box]
        w, h = x2 - x1, y2 - y1
        box_radius = min(w, h) / 2
        sigma = max(min_sigma, box_radius * 5 / (4 * 3))  # 3/5 of gaussian kernel is in box
        cx, cy = round((x1 + x2) / 2), round((y1 + y2) / 2)
        x1, y1, x2, y2 = round(x1), round(y1), round(x2), round(y2)
        density = np.zeros((size, size), dtype=np.float32)
        density[cy, cx] = 100
        density = ndimage.filters.gaussian_filter(density, sigma, mode='constant')
        density_map[0, :, :] += density
        
        ### added forgournd info
        density_map[1, y1:y2, x1:x2] = 1.0 # mark area
        density_map[2, cy, cx] = 1.0     # mark center
        
        ### end of added

    return density_map


def gaussian(kernel):
    sigma = ((kernel - 1) * 0.3 - 1) * 0.3 + 0.8
    s = 2 * (sigma ** 2)
    dx = np.exp(-np.square(np.arange(kernel) - int(kernel / 2)) / s)
    return np.reshape(dx, (-1, 1))
    
    
def generate_density_map_csp(labels, boxes, scale=50.0 / 800, size=50, num_classes=200, min_sigma=1):
    
    num_classes = 3
    density_map = np.zeros((num_classes, size, size), dtype=np.float32)
    for category, box in zip(labels, boxes):
        x1, y1, x2, y2 = [x * scale for x in box]
        x1, y1, x2, y2 = round(x1), round(y1), round(x2), round(y2)
        
        w, h = x2 - x1, y2 - y1
        
        #cx, cy = int((x1 + x2) / 2), int((y1 + y2) / 2)
        dx = gaussian(x2 - x1)
        dy = gaussian(y2 - y1)
        gau_map = np.multiply(dy, np.transpose(dx))
        density_map[0, y1:y2, x1:x2] = np.maximum(density_map[0, y1:y2, x1:x2], gau_map)
        
        #box_radius = min(w, h) / 2
        #sigma = max(min_sigma, box_radius * 5 / (4 * 3))  # 3/5 of gaussian kernel is in box
        cx, cy = round((x1 + x2) / 2), round((y1 + y2) / 2)
        
        #density = np.zeros((size, size), dtype=np.float32)
        #density[cy, cx] = 1
        #density = ndimage.filters.gaussian_filter(density, sigma, mode='constant')
        #density_map[0, :, :] += density
        
        ### added forgournd info
        density_map[1, y1:y2, x1:x2] = 1.0 # mark area
        density_map[2, cy, cx] = 1.0     # mark center
        
        ### end of added

    return density_map