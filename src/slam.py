#!/usr/bin/env python

import sys
sys.path
sys.path.append('/work/repos/LightGlue')

from pathlib import Path
from lightglue import LightGlue, SuperPoint, DISK
from lightglue.utils import load_image, rbd
from lightglue import viz2d
import torch
torch.set_grad_enabled(False);
images = Path('assets')


device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')  # 'mps', 'cpu'

extractor = SuperPoint(max_num_keypoints=2048).eval().to(device)  # load the extractor
matcher = LightGlue(features='superpoint').eval().to(device)

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


# .....

# axes = viz2d.plot_images([image0, image1])
# viz2d.plot_matches(m_kpts0, m_kpts1, color='lime', lw=0.2)
# viz2d.add_text(0, f'Stop after {matches01["stop"]} layers', fs=20)

# kpc0, kpc1 = viz2d.cm_prune(matches01['prune0']), viz2d.cm_prune(matches01['prune1'])
# viz2d.plot_images([image0, image1])
# viz2d.plot_keypoints([kpts0, kpts1], colors=[kpc0, kpc1], ps=10)



