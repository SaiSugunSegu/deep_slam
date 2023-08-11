#!/usr/bin/env python3

import rospy
from std_msgs.msg import String 
import sys
sys.path
sys.path.append('/work/repos/LightGlue')

from pathlib import Path
from lightglue import LightGlue, SuperPoint, DISK
from lightglue.utils import load_image, rbd
from lightglue import viz2d
import torch
torch.set_grad_enabled(False);


def callback(data):
    print("call")
    # load images
    image0 = load_image('/work/repos/LightGlue/assets/DSC_0410.JPG')
    image1 = load_image('/work/repos/LightGlue/assets/DSC_0410.JPG')

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

    
    
def listener():
    rospy.init_node('listener', anonymous=True)
    rospy.Subscriber("/chatter", String, callback)
    print("in node")
    rospy.spin()

if __name__ == '__main__':
    print("in main")
    
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')  # 'mps', 'cpu'
    extractor = SuperPoint(max_num_keypoints=512).eval().to(device)  # load the extractor
    matcher = LightGlue(features='superpoint').eval().to(device)

    listener()