# Third Year Project

## Background

This repository serves as a copy, and possible extension, of my final year project.

An exploration of methods to create a 3D hand-detection pipeline from scratch. The pipeline was created such that it can track a hand in a frame in real time.
<br><br>

The pipeline consists of three stages:
<br>
Hand-detection - detecting if a hand is in an image 
<br>
Keypoint detection - mapping "keypoints" of hand onto an image
<br>
Pose estimation - using the detected keypoints to pose a 3D model
<br><br>

Dissertation report written about the project: [`3D Hand Tracking from an RGB Camera`](Extra_Files\Ryan_Karibwami_Report.pdf)

Short video which briefly describes the project from concept to results: [`Screencast`](Extra_Files\Ryan_Karibwami_Screencast.mp4)

## Code Sourced

Below are the parts of my code which have been directly sourced from other areas, with at most minimal changes, any other code is written by myself. All other code is written directly by me, although they do take inspiration from external sources. 

### BMC Loss

This is used as part of the points detector, as part of its loss, in the following places:

```
points_detector/bmc_loss.py
points_detector/config.py
points_detector/BMC
```

Code is taken from the repository [`github.com/MengHao666/Hand-BMC-pytorch`](https://github.com/MengHao666/Hand-BMC-pytorch), which is an adapted version of the following paper, the code itself is not directly made by the authors of said paper.
```
@article{spurr2020weakly,
  title={Weakly supervised 3d hand pose estimation via biomechanical constraints},
  author={Spurr, Adrian and Iqbal, Umar and Molchanov, Pavlo and Hilliges, Otmar and Kautz, Jan},
  journal={arXiv preprint arXiv:2003.09282},
  volume={8},
  year={2020},
  publisher={Springer}
}
```

### Mano Model Mesh

This is used to create the mesh of the mano model.

The files from the folder `pose_finder/utils/` are directly taken from the repository [`github.com/lmb-freiburg/freihand`](https://github.com/lmb-freiburg/freihand)

### Mano Model Visualisation

This is used to visualise the mesh

The visualisations using open3d are largely adapted from the repository [`github.com/CalciferZh/minimal-hand`](https://github.com/CalciferZh/minimal-hand)
