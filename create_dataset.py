
import os
import json
import argparse
import math
from collections import namedtuple

import numpy as np
import cv2
from tqdm import tqdm
from skimage.io import imread
from shapely.geometry import Polygon, MultiPolygon
import tensorflow as tf
import shapely

from functools import reduce
import operator

# Object Class
Object = namedtuple('Object', 'coord poly cls_idx cls_text')

# Pacth Class
Patch = namedtuple('Patch', 'image_id image row col objects')


def get_patch_image(image, row, col, patch_size):
    patch_image_height = patch_size if image.shape[0] - \
        row > patch_size else image.shape[0] - row
    patch_image_width = patch_size if image.shape[1] - \
        col > patch_size else image.shape[1] - col

    patch_image = image[row: row + patch_image_height,
                        col: col + patch_image_width]

    if patch_image_height < patch_size or patch_image_width < patch_size:
        pad_height = patch_size - patch_image_height
        pad_width = patch_size - patch_image_width
        patch_image = np.pad(
            patch_image, ((0, pad_height), (0, pad_width), (0, 0)), 'constant')

    return patch_image


def load_geojson(filename):
    """ Gets label data from a geojson label file
    :param (str) filename: file path to a geojson label file
    :return: (numpy.ndarray, numpy.ndarray ,numpy.ndarray) coords, chips, and classes corresponding to
            the coordinates, image names, and class codes for each ground truth.
    """

    with open(filename) as f:
        data = json.load(f)

    obj_coords = np.zeros((len(data['features']), 8))
    image_ids = np.zeros((len(data['features'])), dtype='object')
    class_indices = np.zeros((len(data['features'])), dtype=int)
    class_names = np.zeros((len(data['features'])), dtype='object')

    for idx in range(len(data['features'])):
        properties = data['features'][idx]['properties']
        image_ids[idx] = properties['image_id']
        obj_coords[idx] = np.array(
            [float(num) for num in properties['object_imcoords'].split(",")])
        class_indices[idx] = properties['type_id']
        class_names[idx] = properties['type_name']

    return image_ids, obj_coords, class_indices, class_names


def cvt_coords_to_rboxes(coords):
    """ Processes a coordinate array from a geojson into (cy, cx, height, width, theta) format
    :param (numpy.ndarray) coords: an array of shape (N, 8) with 4 corner points of boxes
    :return: (numpy.ndarray) an array of shape (N, 5) with coordinates in proper format
    """

    rboxes = []
    for coord in coords:
        pts = np.reshape(coord, (-1, 2)).astype(dtype=np.float32)
        (cx, cy), (width, height), theta = cv2.minAreaRect(pts)

        if width < height:
            width, height = height, width
            theta += 90.0
        rboxes.append([cy, cx, height, width, math.radians(theta)])

    return np.array(rboxes)


def cvt_coords_to_polys(coords):
    """ Convert a coordinate array from a geojson into Polygons
    :param (numpy.ndarray) coords: an array of shape (N, 8) with 4 corner points of boxes
    :return: (numpy.ndarray) polygons: an array of shapely.geometry.Polygon corresponding to coords
    """

    polygons = []
    for coord in coords:
        polygons.append(
            Polygon([coord[0:2], coord[2:4], coord[4:6], coord[6:8]]))
    return np.array(polygons)


def IoA(poly1, poly2):
    """ Intersection-over-area (ioa) between two boxes poly1 and poly2 is defined as their intersection area over
    box2's area. Note that ioa is not symmetric, that is, IOA(poly1, poly2) != IOA(poly1, poly2).
    :param (shapely.geometry.Polygon) poly1: Polygon1
    :param (shapely.geometry.Polygon) poly2: Polygon2
    :return: (float) IoA between poly1 and poly2
    """
    return poly1.intersection(poly2).area / poly1.area


def tf_int64_feature(value):
    return tf.train.Feature(int64_list=tf.train.Int64List(value=[value]))


def tf_int64_list_feature(value):
    return tf.train.Feature(int64_list=tf.train.Int64List(value=value))


def tf_bytes_feature(value):
    return tf.train.Feature(bytes_list=tf.train.BytesList(value=[value]))


def tf_bytes_list_feature(value):
    return tf.train.Feature(bytes_list=tf.train.BytesList(value=value))


def tf_float_list_feature(value):
    return tf.train.Feature(float_list=tf.train.FloatList(value=value))


def cvt_rbox_to_tfexample(encode_image, image_height, image_width, image_filename, image_format, center_ys, center_xs,
                          heights, widths, thetas, class_texts, class_indices):
    """ Build an Example proto for an example of rbox.
    :param (bytes) encode_image: encoded image
    :param (int) image_height: height of image
    :param (int) image_width: width of image
    :param (bytes) image_filename: encoded image name
    :param (bytes) image_format: encoded file format
    :param (list) center_ys: a list of center y of objects
    :param (list) center_xs: a list of center x of objects
    :param (list) heights: a list of height of objects
    :param (list) widths: a list of width of objects
    :param (list) thetas: a list of theta of objects
    :param (list) class_texts: a list of class text of objects
    :param (list) class_indices: a list of class index of objects
    :return: (tf.train.Example) example proto of rbox
    """

    example = tf.train.Example(features=tf.train.Features(feature={
        'image/height': tf_int64_feature(image_height),
        'image/width': tf_int64_feature(image_width),
        'image/filename': tf_bytes_feature(image_filename),
        'image/encoded': tf_bytes_feature(encode_image),
        'image/format': tf_bytes_feature(image_format),
        'image/object/rbbox/cx': tf_float_list_feature(center_xs),
        'image/object/rbbox/cy': tf_float_list_feature(center_ys),
        'image/object/rbbox/w': tf_float_list_feature(widths),
        'image/object/rbbox/h': tf_float_list_feature(heights),
        'image/object/rbbox/ang': tf_float_list_feature(thetas),
        'image/object/class/text': tf_bytes_list_feature(class_texts),
        'image/object/class/label': tf_int64_list_feature(class_indices),
    }))
    return example


def rotate_box(bb, cx, cy, theta):
    new_bb = list(bb)
    for i, coord in enumerate(bb):
        # opencv calculates standard transformation matrix
        M = cv2.getRotationMatrix2D((cx, cy), theta, 1.0)
        # Prepare the vector to be transformed
        v = [coord[0], coord[1], 1]
        # Perform the actual rotation and return the image
        calculated = np.dot(M, v)
        new_bb[i] = (calculated[0], calculated[1])
    return new_bb


def convert_bbox_to_rbox(cx, cy, h, w, theta):
    x1, y1, x3, y3 = cx-(w/2), cy-(h/2), cx+(w/2), cy+(h/2)
    x4, y4, x2, y2 = cx-(w/2), cy+(h/2), cx+(w/2), cy-(h/2)
    bbox = [(x1, y1), (x2, y2), (x3, y3), (x4, y4)]

    rbox = rotate_box(bbox, cx, cy, -theta)

    rbox = [xx for x in rbox for xx in x]
    rbox = sort_clockwise(rbox)
    rbox = [(x, y) for x, y in zip(rbox[0::2], rbox[1::2])]

    return rbox


def sort_clockwise(points):
    r = list(map(float, points))
    coords = [r[0:2], r[2:4], r[4:6], r[6:8]]
    center = tuple(map(operator.truediv, reduce(
        lambda x, y: map(operator.add, x, y), coords), [len(coords)] * 2))
    sorted_coords = sorted(coords, key=lambda coord: (-135 - math.degrees(
        math.atan2(*tuple(map(operator.sub, coord, center))[::-1]))) % 360)
    return [coord for coords in sorted_coords for coord in coords]


def save_annot(src_dir, what, images, annotations):
    coco_custom_dataset = {
        "info": {
            "description": "Custom Dataset",
            "url": "http://cocodataset.org",
            "version": "1.0",
            "year": 2017,
            "contributor": "Me",
            "date_created": "2020/02/25"
        },
        "licenses": [{
            "url": "http://creativecommons.org/licenses/by-nc-sa/2.0/",
            "id": 1,
            "name": "Attribution-NonCommercial-ShareAlike License"
        },
            {
            "url": "http://creativecommons.org/licenses/by-nc/2.0/",
            "id": 2,
            "name": "Attribution-NonCommercial License"
        }],
        "images": images,
        "annotations": annotations,
        "categories":[{"supercategory": "ship", "id": 1, "name": "small ship"},
                       {"supercategory": "ship", "id": 2, "name": "large ship"},
                       {"supercategory": "ship", "id": 3,
                           "name": "civilian aircraft"},
                       {"supercategory": "ship", "id": 4,
                           "name": "maritime vessels"},
                       {"supercategory": "ship", "id": 5, "name": "small car"},
                       {"supercategory": "ship", "id": 6, "name": "bus"},
                       {"supercategory": "ship", "id": 7, "name": "truck"},
                       {"supercategory": "ship", "id": 8, "name": "train"},
                       {"supercategory": "ship", "id": 9, "name": "crane"},
                       {"supercategory": "ship", "id": 10, "name": "bridge"},
                       {"supercategory": "ship", "id": 11, "name": "oil tank"},
                       {"supercategory": "ship", "id": 12, "name": "dam"},
                       {"supercategory": "ship", "id": 13,
                           "name": "athletic field"},
                       {"supercategory": "ship", "id": 14, "name": "helipad"},
                       {"supercategory": "ship", "id": 15, "name": "roundabout"}]
    }

    if what == "train":
        annt_path = os.path.join(
            src_dir, 'custom_coco_all', 'annotations', 'instances_train2017.json')
    else:
        annt_path = os.path.join(
            src_dir, 'custom_coco_all', 'annotations', 'instances_val2017.json')

    if not(os.path.isdir(os.path.dirname(annt_path))):
        os.makedirs(os.path.dirname(annt_path))

    with open(annt_path, 'w') as f:
        f.write(json.dumps(coco_custom_dataset, indent=4))


image_index = 0
annts_index = 0


def write_tfrecords(src_dir, what, patches):
    """ Write patch information into writer
       :param (str) dst_tfr_path: path to save tfrecords
       :param (list) patches: a list of Patch to save tfrecords
       :param (str) obj_type: object type which is one of {'rbox', 'bbox'}
    """
    global image_index
    global annts_index
    
    images, annotations = [], []
    map_labels = {'small ship': 1,
                  'large ship': 2,
                  'civilian aircraft': 3,
                  'military aircraft': 4,
                  'small car': 5,
                  'bus': 6,
                  'truck': 7,
                  'train': 8,
                  'crane': 9,
                  'bridge': 10,
                  'oil tank': 11,
                  'dam': 12,
                  'athletic field': 13,
                  'helipad': 14,
                  'roundabout': 15,
                  }

    if what == "train":
        image_path = os.path.join(
            src_dir, 'custom_coco_all', 'train2017')
    else:
        image_path = os.path.join(
            src_dir, 'custom_coco_all', 'val2017')

    if not os.path.isdir(image_path):
        os.makedirs(image_path)

    for patch in patches:
        image = cv2.cvtColor(patch.image, cv2.COLOR_RGB2BGR)
        patch_height = patch.image.shape[0]
        patch_width = patch.image.shape[1]
        image_id = patch.image_id.split(".")[0]

        patch_image_path = os.path.join(
            image_path, F"{image_id}_{patch.row}_{patch.col}.png")
        cv2.imwrite(patch_image_path, image)

        images.append({
            "license": 1,
            "file_name": F"{image_id}_{patch.row}_{patch.col}.png",
            "coco_url": "",
            "height": patch_height,
            "width": patch_width,
            "date_captured": "2013-11-14 17:02:52",
            "flickr_url": "",
            "id": image_index
        })

        center_ys, center_xs, heights, widths, thetas, class_indices, class_texts = [
        ], [], [], [], [], [], []
        for coord, poly, cls_idx, cls_text in patch.objects:
            if cls_text == 'etc': continue
            poly = poly.simplify(1.0, preserve_topology=False)            
            polygon = [np.array(poly.exterior.coords).ravel().tolist()]
            
            multi_poly = MultiPolygon([poly])
            x, y, max_x, max_y = multi_poly.bounds
            width = max_x - x
            height = max_y - y
            bbox = (x, y, width, height)
            area = multi_poly.area
            
            annotations.append({
                "segmentation": polygon,
                "area": area,
                "iscrowd": 0,
                "image_id": image_index,
                "bbox": bbox,  # (x, y, width, height)
                "category_id": map_labels[cls_text],
                "id": annts_index
            })
            annts_index += 1
        image_index += 1
        """
            polygon = [(x, y) for x, y in zip(polygon[0::2], polygon[1::2])]
            polygon = np.array(polygon, np.int32)

            bbox = np.array(poly.bounds, np.int32)

            image = image.copy()
            img = cv2.polylines(image, [polygon], True, (0, 255, 0), 2)
            img = cv2.rectangle(
                img, (bbox[0], bbox[1]),  (bbox[2], bbox[3]), (0, 0, 255), 3)

            cv2.imshow('image', img)
            cv2.waitKey(0)
            cv2.destroyAllWindows()
            """
        # trf_writer.write(tfexample.SerializeToString())
    return images, annotations


def create_tfrecords(src_dir, dst_path, patch_size=1024, patch_overlay=384, object_fraction_thresh=0.7,
                     is_include_only_pos=False, what="test"):
    """ Create TF Records from geojson
    :param (str) src_dir: path to a GeoJson file
    :param (str) dst_path: Path to save tfrecords'
    :param (int) patch_size: patch size
    :param (int) patch_overlay: overlay size for patching
    :param (float) object_fraction_thresh: threshold value for determining contained objects
    :param (bool) is_include_only_pos: Whether or not to include only positive patch image(containing at least one object)
    :return:
    """
    images, annotations = [], []
    n_tfrecord = 0

    # Load objects from geojson
    geojson_path = os.path.join(src_dir, 'labels.json')
    image_ids, obj_coords, class_indices, class_names = load_geojson(
        geojson_path)

    obj_polys = cvt_coords_to_polys(obj_coords)
    obj_coords = cvt_coords_to_rboxes(obj_coords)

    if what == 'train':
        datatest = sorted(set(image_ids))
    else:
        datatest = sorted(set(image_ids))[:50]

    # Load image files as TIF
    for image_id in tqdm(datatest):

        image = imread(os.path.join(src_dir, 'images/', image_id))

        # Get data in the current image
        obj_coords_in_image = obj_coords[image_ids == image_id]
        obj_polys_in_image = obj_polys[image_ids == image_id]
        class_indices_in_image = class_indices[image_ids == image_id]
        class_texts_in_image = class_names[image_ids == image_id]

        # Create patches including objects
        patches = []
        step = patch_size - patch_overlay
        for row in range(0, image.shape[0] - patch_overlay, step):
            for col in range(0, image.shape[1] - patch_overlay, step):
                patch_poly = Polygon([(col, row), (col + patch_size, row),
                                      (col + patch_size, row + patch_size), (col, row + patch_size)])

                # Check if a patch contains objects and append objects
                objects_in_patch = []
                for idx, obj_poly in enumerate(obj_polys_in_image):
                    if IoA(obj_poly, patch_poly) > object_fraction_thresh:
                        objects_in_patch.append(Object(obj_coords_in_image[idx], obj_polys_in_image[idx], class_indices_in_image[idx],
                                                       class_texts_in_image[idx]))

                # if a patch contains objects, append the patch to save tfrecords
                if not is_include_only_pos or objects_in_patch:
                    objects_in_patch = [
                        Object(coord=[obj.coord[0] - row, obj.coord[1] - col, obj.coord[2], obj.coord[3], obj.coord[4]],
                               poly=shapely.affinity.translate(obj.poly, xoff=-col, yoff=-row), cls_idx=obj.cls_idx, cls_text=obj.cls_text) for obj in objects_in_patch]
                    patch_image = get_patch_image(image, row, col, patch_size)

                    patches.append(
                        Patch(image_id=image_id, image=patch_image, row=row, col=col, objects=objects_in_patch))

        imgs, annots = write_tfrecords(src_dir, what, patches)
        images.extend(imgs)
        annotations.extend(annots)

        n_tfrecord += len(patches)

    print('N of TFRecords:', n_tfrecord)
    save_annot(src_dir, what, images, annotations)


if __name__ == '__main__':
    parser = argparse.ArgumentParser(
        description='Create TF Records from geojson')
    parser.add_argument('--src_dir',
                        type=str,
                        # required=True,
                        metavar='DIR',
                        default=r"/content/gdrive/My Drive/findShip/",
                        help='Root directory to geojson and images')
    parser.add_argument('--dst_path',
                        type=str,
                        metavar='FILE',
                        default='tfrecords.tfrecords',
                        help='Path to save tfrecords')
    parser.add_argument('--what',
                        type=str,
                        metavar='FILE',
                        default='test',
                        help='Path to save tfrecords')
    parser.add_argument('--patch_size',
                        type=int,
                        default=768,
                        help='Patch size')
    parser.add_argument('--patch_overlay',
                        type=int,
                        default=256,
                        help='Overlay size for patching')
    parser.add_argument('--object_fraction_thresh',
                        type=float,
                        default=0.7,
                        help='Threshold value for determining contained objects')
    parser.add_argument('--is_include_only_pos',
                        dest='is_include_only_pos',
                        # action='store_true',
                        action='store_false',
                        help='Whether or not to include only positive patch image(containing at least one object)')

    args = parser.parse_args()

    create_tfrecords(**vars(args))
