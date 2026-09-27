# Data and large-file policy

The following items are intentionally omitted from the handover copy and from the public-repository preparation:

- KITTI and ModelNet40 datasets;
- PU-Net, PU-GCN, PU-EdgeFormer, PDANS, EAR, TULIP, SPU-PMD, OpenPCDet and CenterPoint third-party checkouts;
- pretrained and adapted model weights (`.pth`, `.pt`, `.ckpt`, TensorFlow checkpoints);
- virtual/Conda environments and compiled CUDA/TensorFlow extensions;
- generated full-validation point-cloud trees;
- raw per-frame detector predictions, caches, logs, and temporary workspaces.

These files should be retained in controlled storage if the chair requires a complete internal archive. They should not be published until the relevant dataset, code, and weight licences have been reviewed.

The repository records compact aggregate results and protocols. Where feasible, future releases should provide a checksum-based manifest and documented download/setup steps instead of redistributing licensed assets.

