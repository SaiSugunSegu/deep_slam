#!/usr/bin/env python3

import rospy
from std_msgs.msg import String 
from sensor_msgs.msg import Image
import matplotlib.pyplot as plt
import message_filters
import sys
import numpy as np
sys.path
sys.path.append('/home/su/work/repos/LightGlue')

from pathlib import Path
from lightglue import LightGlue, SuperPoint, DISK
from lightglue.utils import load_image, rbd
from lightglue import viz2d
import torch
torch.set_grad_enabled(False);


def numpy_image_to_torch(image: np.ndarray) -> torch.Tensor:
    """Normalize the image tensor and reorder the dimensions."""
    if image.ndim == 3:
        image = image.transpose((2, 0, 1))  # HxWxC to CxHxW
    elif image.ndim == 2:
        image = image[None]  # add channel axis
    else:
        raise ValueError(f'Not an image: {image.shape}')
    return torch.tensor(image / 255., dtype=torch.float)

def inter_matcher(image):
    print("")

def match_callback(image0, image1):

    image0 = np.frombuffer(image0.data, dtype=np.uint8).reshape(image0.height, image0.width, -1)
    image1 = np.frombuffer(image1.data, dtype=np.uint8).reshape(image1.height, image1.width, -1)

    # load images
    image0 = numpy_image_to_torch(image0).cuda()
    image1 = numpy_image_to_torch(image1).cuda()

    # extract features using 'Super Point' with max_num_key_point
    feats0 = extractor.extract(image0.to(device))
    feats1 = extractor.extract(image1.to(device))

    # match feature using 'Light Glue'  
    matches01 = matcher({'image0': feats0, 'image1': feats1})
    feats0, feats1, matches01 = [rbd(x) for x in [feats0, feats1, matches01]]  # remove batch dimension

    # Keypoints in Image0, Keypoints in Image1, Match Points in Images
    kpts0, kpts1, matches = feats0['keypoints'], feats1['keypoints'], matches01['matches']

    # Match points in Image0, Image1
    m_kpts0, m_kpts1 = kpts0[matches[..., 0]], kpts1[matches[..., 1]]
    print("Match Points", m_kpts0.shape[0])

    axes = viz2d.plot_images([image0, image1])
    viz2d.plot_matches(m_kpts0, m_kpts1, color='lime', lw=0.2)
    viz2d.add_text(0, f'Stop after {matches01["stop"]} layers', fs=20)

    # plt.imshow(image0)

    
def listener():
    rospy.init_node('listener', anonymous=True)

    # rospy.Subscriber("/cam_front/raw", Image, inter_matcher)
    # rospy.Subscriber("/cam_front_right/raw", Image, inter_matcher)
    # rospy.Subscriber("/cam_front_left/raw", Image, inter_matcher)

    front_sub = message_filters.Subscriber("/cam_front/raw", Image)
    right_sub = message_filters.Subscriber("/cam_front_right/raw", Image)
    left_sub = message_filters.Subscriber("/cam_front_left/raw", Image)

    front_right = message_filters.ApproximateTimeSynchronizer([front_sub, right_sub], 10, 0.1, allow_headerless=True)
    front_right.registerCallback(match_callback)

    front_left = message_filters.ApproximateTimeSynchronizer([front_sub, left_sub], 10, 0.1, allow_headerless=True)
    front_left.registerCallback(match_callback)

    rospy.spin()

if __name__ == '__main__':
    
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')  # 'mps', 'cpu'
    extractor = SuperPoint(max_num_keypoints=2048).eval().to(device).cuda()  # load the extractor
    matcher = LightGlue(features='superpoint').eval().to(device).cuda()

    listener()