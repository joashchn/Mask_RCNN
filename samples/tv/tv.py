"""
Mask R-CNN
Train on the toy tv dataset and implement color splash effect.

Copyright (c) 2018 Matterport, Inc.
Licensed under the MIT License (see LICENSE for details)
Written by Waleed Abdulla

------------------------------------------------------------

Usage: import the module (see Jupyter notebooks for examples), or run from
       the command line as such:

    # Train a new model starting from pre-trained COCO weights
    python3 logo.py train --dataset=/path/to/tv/dataset --weights=coco

    # Resume training a model that you had trained earlier
    python3 logo.py train --dataset=/path/to/tv/dataset --weights=last

    # Train a new model starting from ImageNet weights
    python3 logo.py train --dataset=/path/to/tv/dataset --weights=imagenet

    # Apply color splash to an image
    python3 logo.py splash --weights=/path/to/weights/file.h5 --image=<URL or path to file>

    # Apply color splash to video using the last weights you trained
    python3 logo.py splash --weights=last --video=<URL or path to file>
"""

import os
import sys
import json
import datetime
import numpy as np
import skimage.draw
from skimage import io, data_dir
import cv2

# Root directory of the project
ROOT_DIR = os.path.abspath("../../")

# Import Mask RCNN
sys.path.append(ROOT_DIR)  # To find local version of the library
from mrcnn.config import Config
from mrcnn import model as modellib, utils

# Path to trained weights file
COCO_WEIGHTS_PATH = os.path.join(ROOT_DIR, "mask_rcnn_coco.h5")

# Directory to save logs and model checkpoints, if not provided
# through the command line argument --logs
DEFAULT_LOGS_DIR = os.path.join(ROOT_DIR, "logs")


############################################################
#  Configurations
############################################################

def img_variance(img):
    # img = cv2.imread(path)
    v_b = img[:, :, 0].std()
    v_g = img[:, :, 1].std()
    v_r = img[:, :, 2].std()

    return v_b, v_g, v_r

class TvConfig(Config):
    """Configuration for training on the toy  dataset.
    Derives from the base Config class and overrides some values.
    """
    # Give the configuration a recognizable name
    NAME = "tv"

    # We use a GPU with 12GB memory, which can fit two images.
    # Adjust down if you use a smaller GPU.
    IMAGES_PER_GPU = 2

    # Number of classes (including background)
    NUM_CLASSES = 1 + 1  # Background + tv

    # Number of training steps per epoch
    STEPS_PER_EPOCH = 100

    # Skip detections with < 90% confidence
    DETECTION_MIN_CONFIDENCE = 0.9


############################################################
#  Dataset
############################################################

class TvDataset(utils.Dataset):

    def load_tv(self, dataset_dir, subset):
        """Load a subset of the tv dataset.
        dataset_dir: Root directory of the dataset.
        subset: Subset to load: train or val
        """
        # Add classes. We have only one class to add.
        self.add_class("tv", 1, "tv")

        # Train or validation dataset?
        assert subset in ["train", "val"]
        dataset_dir = os.path.join(dataset_dir, subset)

        # Load annotations
        # VGG Image Annotator (up to version 1.6) saves each image in the form:
        # { 'filename': '28503151_5b5b7ec140_b.jpg',
        #   'regions': {
        #       '0': {
        #           'region_attributes': {},
        #           'shape_attributes': {
        #               'all_points_x': [...],
        #               'all_points_y': [...],
        #               'name': 'polygon'}},
        #       ... more regions ...
        #   },
        #   'size': 100202
        # }
        # We mostly care about the x and y coordinates of each region
        # Note: In VIA 2.0, regions was changed from a dict to a list.
        annotations = json.load(open(os.path.join(dataset_dir, "train_tv.json")))
        annotations = list(annotations.values())  # don't need the dict keys
        # print(annotations)

        # The VIA tool saves images in the JSON even if they don't have any
        # annotations. Skip unannotated images.
        # annotations = [a for a in annotations if a['shapes']][0]

        # Add images
        for a in annotations:
            #     # Get the x, y coordinaets of points of the polygons that make up
            #     # the outline of each object instance. These are stores in the
            #     # shape_attributes (see json format above)
            #     # The if condition is needed to support VIA versions 1.x and 2.x.
            #     if type(a['shapes']) is dict:
            #         polygons = [r['shape_attributes'] for r in a['regions'].values()]
            #     else:
            #         polygons = [r['shape_attributes'] for r in a['regions']]

            # print(np.array(a['shapes'][0]['points']))
            # print(np.array(a['shapes'][0]['points'])[:, 0].tolist())

            polygons = [
                {'name': a['shapes'][0]['label'], 'all_points_x': np.array(a['shapes'][0]['points'])[:, 0].tolist(),
                 'all_points_y': np.array(a['shapes'][0]['points'])[:, 1].tolist()}]

            # load_mask() needs the image size to convert polygons to masks.
            # Unfortunately, VIA doesn't include it in JSON, so we must read
            # the image. This is only managable since the dataset is tiny.
            image_path = os.path.join(dataset_dir, a['imagePath'])
            image = skimage.io.imread(image_path)
            height, width = image.shape[:2]

            self.add_image(
                "tv",
                image_id=a['imagePath'],  # use file name as a unique image id
                path=image_path,
                width=width, height=height,
                polygons=polygons)

    def load_mask(self, image_id):
        """Generate instance masks for an image.
       Returns:
        masks: A bool array of shape [height, width, instance count] with
            one mask per instance.
        class_ids: a 1D array of class IDs of the instance masks.
        """
        # If not a tv dataset image, delegate to parent class.
        image_info = self.image_info[image_id]
        if image_info["source"] != "tv":
            return super(self.__class__, self).load_mask(image_id)

        # Convert polygons to a bitmap mask of shape
        # [height, width, instance_count]
        info = self.image_info[image_id]
        mask = np.zeros([info["height"], info["width"], len(info["polygons"])],
                        dtype=np.uint8)
        for i, p in enumerate(info["polygons"]):
            # Get indexes of pixels inside the polygon and set them to 1
            rr, cc = skimage.draw.polygon(p['all_points_y'], p['all_points_x'])
            mask[rr, cc, i] = 1

        # Return mask, and array of class IDs of each instance. Since we have
        # one class ID only, we return an array of 1s
        return mask.astype(np.bool), np.ones([mask.shape[-1]], dtype=np.int32)

    def image_reference(self, image_id):
        """Return the path of the image."""
        info = self.image_info[image_id]
        if info["source"] == "tv":
            return info["path"]
        else:
            super(self.__class__, self).image_reference(image_id)


def train(model):
    """Train the model."""
    # Training dataset.
    dataset_train = TvDataset()
    dataset_train.load_tv(args.dataset, "train")
    dataset_train.prepare()

    # Validation dataset
    dataset_val = TvDataset()
    dataset_val.load_tv(args.dataset, "val")
    dataset_val.prepare()

    # *** This training schedule is an example. Update to your needs ***
    # Since we're using a very small dataset, and starting from
    # COCO trained weights, we don't need to train too long. Also,
    # no need to train all layers, just the heads should do it.
    print("Training network heads")
    model.train(dataset_train, dataset_val,
                learning_rate=config.LEARNING_RATE,
                epochs=30,
                layers='heads')


def color_splash(image, mask):
    """Apply color splash effect.
    image: RGB image [height, width, 3]
    mask: instance segmentation mask [height, width, instance count]

    Returns result image.
    """
    # Make a grayscale copy of the image. The grayscale copy still
    # has 3 RGB channels, though.
    gray = skimage.color.gray2rgb(skimage.color.rgb2gray(image)) * 255
    # Copy color pixels from the original color image where mask is set
    if mask.shape[-1] > 0:
        # We're treating all instances as one, so collapse the mask into one layer
        mask = (np.sum(mask, -1, keepdims=True) >= 1)
        splash = np.where(mask, image, gray).astype(np.uint8)
    else:
        splash = gray.astype(np.uint8)
    return splash


def detect_and_color_splash(model, image_path=None, video_path=None):
    # assert image_path or video_path

    # Image or video?
    print(len(image_path))
    if image_path.any():
        import cv2
        import time
        # Run model detection and generate the color splash effect
        # print("Running on {}".format(args.image))
        # # Read image
        # image = skimage.io.imread(args.image)

        # print("Running on {}".format(image_path))
        r = model.detect([image_path], verbose=1)
        # t = time.time()
        # print((int(round(t * 1000))))
        # Color splash
        region = []
        region_type = []
        for i in range(len(r)):
            index = np.argwhere(r[i]['masks'] == True)
            y_min = np.min(index[:, 0])
            x_mid = int((np.min(index[:, 1]) + np.max(index[:, 1]))/2)
            # splash = color_splash(image_path[i], r[i]['masks'])
            # splash = cv2.rectangle(splash, (x_mid-5, y_min-5), (x_mid+5, y_min-15), (0, 0, 255), 2)
            if x_mid-5>0 and y_min-15>0 and x_mid+5<300:
                region.append(image_path[i][y_min-15:y_min-5, x_mid-5:x_mid+5])
                region_type.append(0)
            else:
                region.append(image_path[i])
                region_type.append(1)

            # Save output
            # file_name = "detect_"+str(i)+".png"
            # skimage.io.imsave('splash_'+file_name, splash)
            # skimage.io.imsave('region_'+file_name, region)

        # cv2.imshow('region', region)
        # cv2.waitKey(2)

        return region, region_type
    elif video_path.any():
        import cv2
        # Video capture
        vcapture = cv2.VideoCapture(video_path)
        width = int(vcapture.get(cv2.CAP_PROP_FRAME_WIDTH))
        height = int(vcapture.get(cv2.CAP_PROP_FRAME_HEIGHT))
        fps = vcapture.get(cv2.CAP_PROP_FPS)

        # Define codec and create video writer
        file_name = "splash_{:%Y%m%dT%H%M%S}.avi".format(datetime.datetime.now())
        vwriter = cv2.VideoWriter(file_name,
                                  cv2.VideoWriter_fourcc(*'MJPG'),
                                  fps, (width, height))

        count = 0
        success = True
        while success:
            print("frame: ", count)
            # Read next image
            success, image = vcapture.read()
            if success:
                # OpenCV returns images as BGR, convert to RGB
                image = image[..., ::-1]
                # Detect objects
                r = model.detect([image], verbose=0)[0]
                # Color splash
                splash = color_splash(image, r['masks'])
                # RGB -> BGR to save image to video
                splash = splash[..., ::-1]
                # Add image to video writer
                vwriter.write(splash)
                count += 1
        vwriter.release()
    print("Saved to ", file_name)


############################################################
#  Training
############################################################

# if __name__ == '__main__':
#     import argparse
#
#     # Parse command line arguments
#     parser = argparse.ArgumentParser(
#         description='Train Mask R-CNN to detect tvs.')
#     parser.add_argument("command",
#                         metavar="<command>",
#                         help="'train' or 'splash'")
#     parser.add_argument('--dataset', required=False,
#                         metavar="/path/to/tv/dataset/",
#                         help='Directory of the tv dataset')
#     parser.add_argument('--weights', required=True,
#                         metavar="/path/to/weights.h5",
#                         help="Path to weights .h5 file or 'coco'")
#     parser.add_argument('--logs', required=False,
#                         default=DEFAULT_LOGS_DIR,
#                         metavar="/path/to/logs/",
#                         help='Logs and checkpoints directory (default=logs/)')
#     parser.add_argument('--image', required=False,
#                         metavar="path or URL to image",
#                         help='Image to apply the color splash effect on')
#     parser.add_argument('--video', required=False,
#                         metavar="path or URL to video",
#                         help='Video to apply the color splash effect on')
#     args = parser.parse_args()
#
#     # Validate arguments
#     if args.command == "train":
#         assert args.dataset, "Argument --dataset is required for training"
#     elif args.command == "splash":
#         assert args.image or args.video, \
#             "Provide --image or --video to apply color splash"
#
#     print("Weights: ", args.weights)
#     print("Dataset: ", args.dataset)
#     print("Logs: ", args.logs)
#
#     # Configurations
#     if args.command == "train":
#         config = TvConfig()
#     else:
#         class InferenceConfig(TvConfig):
#             # Set batch size to 1 since we'll be running inference on
#             # one image at a time. Batch size = GPU_COUNT * IMAGES_PER_GPU
#             GPU_COUNT = 1
#             IMAGES_PER_GPU = 1
#
#
#         config = InferenceConfig()
#     config.display()
#
#     # Create model
#     if args.command == "train":
#         model = modellib.MaskRCNN(mode="training", config=config,
#                                   model_dir=args.logs)
#     else:
#         model = modellib.MaskRCNN(mode="inference", config=config,
#                                   model_dir=args.logs)
#
#     # Select weights file to load
#     if args.weights.lower() == "coco":
#         weights_path = COCO_WEIGHTS_PATH
#         # Download weights file
#         if not os.path.exists(weights_path):
#             utils.download_trained_weights(weights_path)
#     elif args.weights.lower() == "last":
#         # Find last trained weights
#         weights_path = model.find_last()
#     elif args.weights.lower() == "imagenet":
#         # Start from ImageNet trained weights
#         weights_path = model.get_imagenet_weights()
#     else:
#         weights_path = args.weights
#
#     # Load weights
#     print("Loading weights ", weights_path)
#     if args.weights.lower() == "coco":
#         # Exclude the last layers because they require a matching
#         # number of classes
#         model.load_weights(weights_path, by_name=True, exclude=[
#             "mrcnn_class_logits", "mrcnn_bbox_fc",
#             "mrcnn_bbox", "mrcnn_mask"])
#     else:
#         model.load_weights(weights_path, by_name=True)
#
#     # Train or evaluate
#     if args.command == "train":
#         train(model)
#     elif args.command == "splash":
#         path_ = '/Users/joash/PycharmProjects/Mask_RCNN/imagess/'
#         for f in os.listdir(path_):
#             if 'jpg' in f:
#                 # 1、得到region
#                 region, region_type = detect_and_color_splash(model, image_path=path_ + f,
#                                         video_path=args.video)
#                 # 2、计算方差
#                 v_b, v_g, v_r = img_variance(region)
#                 if v_b <10 and v_g<10 and v_r<10:
#                     variance_type = 0
#                 else:
#                     variance_type = 1
#                 # 3、转lab，计算差距
#                 img_t = cv2.cvtColor(region, cv2.COLOR_BGR2LAB)
#                 img_t1 = img_t[:, :, 0]
#                 img_t2 = img_t[:, :, 1]
#                 img_t3 = img_t[:, :, 2]
#
#                 t1 = np.mean(img_t1) / 255.
#                 t2 = np.mean(img_t2) / 255.
#                 t3 = np.mean(img_t3) / 255.
#                 x = [t1, t2, t3]
#                 print(region_type)
#                 print(variance_type)
#                 print(x)
#
#
#     else:
#         print("'{}' is not recognized. "
#               "Use 'train' or 'splash'".format(args.command))

def test_process():
    import argparse

    parser = argparse.ArgumentParser(
        description='Train Mask R-CNN to detect tvs.')
    parser.add_argument('--logs', required=False,
                        default=DEFAULT_LOGS_DIR,
                        metavar="/path/to/logs/",
                        help='Logs and checkpoints directory (default=logs/)')
    args = parser.parse_args()

    class InferenceConfig(TvConfig):
        # Set batch size to 1 since we'll be running inference on
        # one image at a time. Batch size = GPU_COUNT * IMAGES_PER_GPU
        GPU_COUNT = 1
        IMAGES_PER_GPU = 1
        WEIGHT_PATH = '/Users/joash/PycharmProjects/Mask_RCNN/samples/mask_rcnn_tv_0030.h5'
        IMG_PATH = '/Users/joash/PycharmProjects/Mask_RCNN/imagess/'
    config = InferenceConfig()
    config.display()
    model = modellib.MaskRCNN(mode="inference", config=config, model_dir=args.logs)
    model.load_weights(config.WEIGHT_PATH, by_name=True)

    # image_ = io.ImageCollection(str(config.IMG_PATH + '/*.jpg'))
    for f in os.listdir(config.IMG_PATH):
        print(config.IMG_PATH + f)
        image_ = skimage.io.imread(config.IMG_PATH + f)
        # print(image_)
        region, region_type = detect_and_color_splash(model, image_path=image_)

    # img_arr = []
    # for i in range(len(image_)):
    #     img_data = io.imread(image_[i])
    #     print(img_data)
    #     img_arr.append(img_data)
    #     print(img_arr[i, :, :])
    #     # region, region_type = detect_and_color_splash(model, image_path=config.IMG_PATH+f)
    #     t = time.time()
    #     print((int(round(t * 1000))))
    # print(region, region_type)


if __name__ == '__main__':
    test_process()