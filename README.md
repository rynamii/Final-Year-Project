# Third Year Project



## Code Sourced

Below are the parts of my code which have been directly sourced from other areas, with at most minimal changes, any other code is written by myself. All other code is written directly by me, although they do take inspiration from external sources. 

### BMC Loss

This is used as part of the points detector, as part of its loss, in the following places:

```
points_detector/bmc_loss.py
points_detector/config.py
points_detector/BMC
```

Code is take from the repository [`github.com/MengHao666/Hand-BMC-pytorch`](https://github.com/MengHao666/Hand-BMC-pytorch), which is an adapted version of the following paper, the code itself is not directly made by the authors of said paper.
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