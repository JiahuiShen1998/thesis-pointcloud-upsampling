# Reproducibility notes

The experiment workspace spans legacy TensorFlow/CUDA point-cloud operators, PyTorch PointRCNN, OpenPCDet/CenterPoint, and method-specific environments. A single unverified `requirements.txt` would be misleading, so the package preserves code and protocol evidence without claiming that one environment specification reproduces every branch.

For a future clean release, pin separate containers or Conda environments for:

1. PU-Net / PU-GCN-family TensorFlow inference;
2. PointRCNN inference and adaptation;
3. OpenPCDet / CenterPoint inference and adaptation;
4. PDANS and other PyTorch upsampling methods;
5. reporting and figure generation.

Each environment should record the Python, framework, CUDA, compiler, GPU, and custom-op revisions actually tested.

