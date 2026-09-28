# PU-GCN = detector adaptation: complete protocol, point selection and code description

## 0. Direct answers to two questions

**"3 epochs" means detector adaptation, not PU-GCN training.**PointRCNN is RPN 3 epochs, then offline RCNN 3 epochs;CenterPoint is 3 epochs.PU-GCN no retrained on KITTI by 3 epochs, this experiment fixed by author PU1K `model-100`. The code only proves that 3 is a fixed adaptation length for pre-written scripts. The record "validation convergence Curve Selection 3" cannot be found.Only epoch 3 has been saved and validation AP of no epoch 1/2, so cannot claims to have converge;Can only say three rounds are completed and training loss drops.

**3N/3M is not the three new sets of points that PU-GCN has identified itself.**PU-GCN for each anchor output 4 residual coordinates; Overlap  patch  After merging, use fixed  seed  uniformly without replacement  sampling  Strictly enforced.  4N/4M. When constructing observed-first input, keep N/M observed rows in its entirety and use another fixed seed sample uniformly without replacement 3N/3M from strict 4N/4M forecast rows.The current no is re-electing 3N/3M at the distance from confidence to observed, FPS, curvature or voxel.

The three points must be distinguished:

| Phase | Candidates Results | Practical guidelines |
|---|---|---|
| patch Input Selection | All frame N/M  /  Multiple 2048-point patch | density threshold + seeded first center + XYZ FPS + radius/kNN coverage |
| strict 4× | All patch raw Output `J×8192` → 4N/4M | `np.random.default_rng(20260702+frame_id).choice(..., replace=False)` |
| observed-first Add Point | strict predicted 4N/4M → 3N/3M | SHA-256 was born after frame seed `choice(..., replace=False)`; Not related to geometric and score |

## 1..

1. **PU-GCN no only trains 3 epochs.**This experiment no retrain PU-GCN on KITTI, but fixed uses the author's PU1K checkpoint `model-100`. The original training parameters for checkpoint are: `max_epochs: 200`, checkpoint pointer actually pointed `model-100`.
2. **3 epochs is detector adaptation.**PointRCNN  It's...  RPN  Training  3 epochs,  Offline after  RCNN  Retrain.  3 epochs; CenterPoint adaptation Training 3 epochs.The initialization weights are the official PointRCNN and the official 80-epoch CenterPoint KITTI checkpoint respectively.
3. **The old detector number comes only from pilot256.**Although adaptation uses the full 3712 KITTI train frame, AP evaluation only 256 validation frame.It cannot answers: "Is the whole validation running by?"This directory regenerate and assess all **3769** A validation frame;`3796` It's an error.

Data boundary reads as follows:

| Purpose | split | Actual non-empty only frame | Whether to use label |
|---|---|---:|---|
| detector adaptation | KITTI train | 3712 | Yes, only training |
| Finally, validation | KITTI val | 3769 | only for final KITTI AP calculation |

## 2. Line A, Line B and two detector inputs

So one, frame, the original KITTI point cloud, is

\[
O_i\in\mathbb{R}^{N_i\times4},\quad (x,y,z,r)
\]

of which `r` It's intensity.

### Line A

- observed Input: Original point cloud `O_i`point count `N_i`;
- PU-GCN direct Output: Strict `4N_i` Points;
- observed-first detector input:`N_i` Original observed dot, plus from direct `4N_i` Medium without replacement `3N_i` A web-based output point, which ultimately remains `4N_i`.

The formula is:

\[
D^A_i=\operatorname{StrictSample}(\operatorname{Merge}(\operatorname{PUGCN}(P^A_{ij})),4N_i)
\]

\[
X^A_i=O_i\;\Vert\;D^A_i[S_i],\qquad |S_i|=3N_i
\]

### Line B

Start with the original. `N_i` downsampling:

\[
M_i=\lfloor N_i/4\rfloor
\]

- `M_i` A row index by NumPy `Generator.choice(N_i, M_i, replace=False)` Access;
- seed is `20260702 + int(frame_id)`;
- index is sorted and then indexed, so the relative order of the original document is retained;
- validation manifest Audited 3769/3769 frame, all satisfied `M=floor(N/4)`.

Subsequently:

\[
D^B_i=\operatorname{StrictSample}(\operatorname{Merge}(\operatorname{PUGCN}(P^B_{ij})),4M_i)
\]

\[
X^B_i=B_i\;\Vert\;D^B_i[T_i],\qquad |T_i|=3M_i
\]

That's why `4M_i=4 floor(N_i/4)=N_i-(N_i mod 4)`It's very close, but not necessarily equal. `N_i`.

## 3. 3N / 3M How did you get it?

It is not an "Additional Three Point" label returned directly from the original PU-GCN network, but a second determinative sample of detector input into the application of the suitable layer:

```python
expected = 4 * observed.shape[0]
assert predicted.shape[0] == expected
seed = stable_seed(20260718, "e1", reference_token, frame_id)
rng = np.random.default_rng(seed)
selected = rng.choice(expected, size=3 * observed.shape[0], replace=False)
detector_input = np.concatenate((observed, predicted[selected]), axis=0)
```

`stable_seed` It is the former 8 bytes of SHA-256 digest after the string is spelled, and it is amputated into 32-bit integer.Line A `reference_token` Yes. `original_x4_pu_gcn`; Line B `downsampled_x4_pu_gcn`.

Here'predicted/generated 'is**Source definition**: the row from PU-GCN output.It is not a collective difference definition.Accomplishing no coordinates heavy, and no deletes the output that happens to be near observed. Therefore, cannot claims  3N/3M  On geometric  observed  Different.

## 4. A whole frame How to select 2048 point patch Enter PU-GCN

Use of this experiment `fps_ball_cover_knn_v3`Every patch fixed `K=2048`, Line A and Line B parameters are:

| Parameters | Line A | Line B |
|---|---:|---:|
| primary patch budget | `ceil(N/2048) × 1` | `ceil(M/2048) × 1` |
| primary ball radius | 2 m | 4 m |
| primary center Minimum Support Point | 2048 | 2048 |
| cover radius | 6 m | 6 m |
| cover eligibility | radius 6 m at least 32 points | Same Left |
| supplemental kNN floor | 2048 unique points | Same Left |
| center/FPS Distance | XYZ Euro-range, unit m | Same Left |
| seed | `20260702 + frame_id` | Same Left |

Specific order:

1. Use it. `cKDTree` Statistics the support of point count for each input point within primary radius, allowing only support of point count `>=2048` The point becomes primary center.
2. Select the first center in eligible center with seeded uniform.
3. Follow-up center uses standard FPS: Select to the largest eligible point of the minimum square distance of the already existing center collection.
4. For every primary center do radius query, press `(distance, original_index)` Stable sort, take before 2048.Since center has requested at least 2048 points in radius, primary patch does not need a copy fill.
5. The statistical entry point appears in any patch.For points in radius 6 m that have at least 32 points but are not yet covered, select the uncovered point "nearest distance from existing centers" as supplemental center.
6. supplemental patch places uncovered 6 m ball points and already covered 6 m ball points;Still below 2048, replace 2048 unique rows with the full frame kNN of center.
7. The cover-eligible input point requires 100% coverage.An isolated point of less than 32 in radius 6 m is not covered by the 100% guarantee;protocol will not falsify these points to claim full coverage.

All patch read XYZ only, not jitter, unplugged, unmodified.Every frame metadata keeps center for each patch, original row index, support point count, coverage and geometric diagnosis.

## 5. PU-GCN, WHAT DO YOU DO ON EVERY patch?

Pre-training parameters:

- dataset: PU1K;
- model: PUGCN;
- training patch: 256 → 1024;
- inference patch: This protocol was converted into 2048  /  8192 using a full volume/chart network;
- block: Inception DenseGCN;
- `n_blocks=2`, `channels=32`, `k=20`, `d=2`;
- upsampler: NodeShuffle;
- ratio: 4;
- fixed checkpoint: `model-100`.

Each 2048-point patch will start with a central and unit ball. normalization:

\[
\hat p=(p-c)/r_{max}
\]

The network predicts residual at normalization coordinates.Original achieves the input anchor tile 4 times and adds the forecast offset to the corresponding anchor:

\[
q_{j,t}=p_j+\Delta_{j,t},\quad t=1,2,3,4
\]

Final re-implementation `q = centroid + q_hat * furthest_distance` Revert to LiDAR coordinates.So each 2048 patch returns 8192 network output row.2 m of Line A and 4 m of Line B are the absolute spatial ranges constructed by patch;Once sent into the network, each patch will still be normalization.

## 6. gets strict 4N / 4M from overlapping patch output

Suppose a frame eventually has `J_i` patch, raw candidate:

\[
R_i=J_i\times8192
\]

runner  Press  patch ID  Order  concatenate  All  raw output. Then strict adapter used seed `20260702 + frame_id`:

- If `R_i == 4N_i`(or) `4M_i`) direct copying;
- If `R_i > target`From `R_i` Line even, without replacement extract exact target;
- If `R_i < target`— immediate failure;It is prohibited to copy input points, repeat raw row or make up false points;
- intensity does not be predicted by PU-GCN, but does 1-NN from strict XYZ, the most recent observed XYZ;
- Final Save Standard KITTI float32 XYZI `.bin`.

Overlap patch may produce multiple sets of outputs for the same input anchor;The current protocol does not do voxel/grid/ coordinates to weigh.This is part of protocol, cannot replaces it after the results come out.

## 7. detector adaptation protocol

### PointRCNN

- official init: `tools/PointRCNN.pth`;
- train split: full 3712 frame;
- Each input arm independent adaptation;
- RPN: 3 epochs, batch size 1, workers 0;
- Export epoch-3 RPN train split proposals and features;
- Merge adapted RPN and official RCNN as the initialization of offline RCNN;
- offline RCNN: 3 epochs, batch size 1, workers 0;
- optimizer: Adam one-cycle;
- LR: 0.0002; LR clip: 1e-6; decay step `[2]`; No LR warmup;
- seed: 20260823; cuDNN deterministic;
- GT database augmentation: Closed;
- Save checkpoint only in epoch 3.

PointRCNN  It's time for separate training.  direct  with  observed-first  It's...  matched checkpoint; Line B baseline also has an independent adapted checkpoint.

### CenterPoint

- official init: KITTI 80-epoch checkpoint;
- train split: full 3712 frame;
- batch size 2, workers 0;
- epochs: 3;
- optimizer: Adam one-cycle;
- LR: 0.0003; weight decay 0.01;
- seed: 666;
- GT sampling: Closed;Other world flip/rotation/scaling reservations;
- checkpoint interval: 3, so only epoch 3 is saved;
- PU-GCN adaptation training input is observed-first, not generated-only/direct.

## 8. Why 3 epochs, Is converge

The existing record supports the conclusion that:**3 epochs is the pre-code adaptation length of fixed, and no selects epoch records based on validation convergence.**The historical script no records why 3 was chosen instead of 6 or 10, so cannot wrote the assumptions about the calculation of the budget as a proven experimental basis.

PointRCNN  Every one  arm  It's...  epoch  Internal  loss  The medians are down; Line A direct RPN from 1.6241 to 1.3475, RCNN from 1.0430 to 0.9596.CenterPoint Line A mean loss from 2.33 to 2.11, Line B PU-GCN from 3.69 to 3.10.Full by arm `convergence_audit.md`.

However, cannot therefore claims that converge:

- Only epoch 3, no epoch 1, 2 checkpoint;
- no in training by epoch full-val AP
- loss is still declining and does not show plateau;
- training loss Fall does not imply detector validation AP has been to the best.

"3-epoch adaptation is normal and optimization loss is down;Whether convergence no is currently certified by checkpoint schedule.“If a paper must be written convergence, at least 6–10 epochs and every epoch must be saved and judged by a pre-declared held-out selection split or by epoch full-val;cannot used the final test/val result to pick epoch and still claim no selection bias.

## 9. detector, WHAT REAL READING

### PointRCNN

frame-level `.bin` Press camera FOV and configure range filter first.RPN and fixed Enter 16384 point:

- If the candidate is more than 16384, usually keep all depths `>=40 m` far points, supplemented by near points without replacement;
- If far points itself `>=16384`, the audited sampler-safe path is changed from full candidate sample uniformly without replacement 16384 to avoid negative sample size;
- If the candidate is less than 16384, retain all and draw additional row;
- full-val runner  In each of the pickups  frame  Reset  NumPy seed  Yes.  `(20260908 + int(frame_id)) & 0xffffffff`, so that the same input is selected below the two weights of official/adapted to the same set of points.Original `eval_rcnn.py` The seed will be reset inside the function, so this time the frame seed will be implemented in the dataset take-point function;Every frame saves actual 16384-point tensor SHA-256 and shape.

### CenterPoint

test configuration does not do fixed-count point sampling.It retains FOV and is located `[0,-40,-3,70.4,40,1]` , and then to voxel size `[0.05,0.05,0.1]` voxelize; For each voxel up to 5 points and test up to 40000 voxels.test does not shuffle.

Therefore, the concept of “constraint 4N/4M” in the document is not the same as that of “effective points/ voxel in the actual size of the network”, which should be indicated in the report.

## 10. New Full validation Comparative Matrix

Every arm must meet `status=PASS` and `frame_count=3769` To enter the final table. Main indicator is:  KITTI Car AP_R40  It's...  3D/BEV easy, moderate, hard; CenterPoint also keeps Pedestrian and Cyclist results.

Special note: Local PointRCNN History evaluator `get_mAP` Default to take 41 precision samples every 4 times, get **AP_R11**. New runner Rear `precision[..., 1:41]` mean, i.e. **AP_R40**, with the local OpenPCDet `get_mAP_R40` The definition is consistent. Old  PointRCNN pilot  Numbers need to be original.  evaluator  Caliber label, cannot, directly known as  AP_R40  Or with  CenterPoint R40  Combining the same indicator.

plan Matrix:

- PointRCNN: 11 arms, including A baseline, A direct official/adapted, A observed-first official/adapted, B baseline official/adapted, B direct official/adapted, B observed-first official/adapted;
- CenterPoint: 9 arms, including A/B baseline official (and B baseline adapted), A/B direct official, A/B observed-first official/adapted.

Real-time summary under implementation `full_val_detector_matrix.md`. The old pilot256 AP will not be written into this full-val table.

## 11. Original code and recurrence entrance

### Original author PU-GCN code/parameter

- `external/PU-GCN/Upsampling/generator.py`: PUGCN graph, NodeShuffle, coordinate residual;
- `external/PU-GCN/Upsampling/model.py`: patch normalization/inverse normalization;
- `external/PU-GCN/Common/ops.py`: feature extractor and up-unit;
- `external/PU-GCN/tf_lib/gcn_lib/vertex.py`: Inception DenseGCN / NodeShuffle;
- `external/PU-GCN/pretrained/pu1k-pugcn/args.txt`Full training parameters for the author checkpoint;
- `external/PU-GCN/pretrained/pu1k-pugcn/checkpoint`: Actual checkpoint pointer.

### This experiment protocol Code

- `scripts/prepare_kitti_downsampled_x4_val.py`: Line B `M=floor(N/4)`;
- `scripts/wrappers/kitti_patch_extractor.py`: 2048 point patch selection;
- `scripts/wrappers/tf_pugcn_family_patch_infer_many.py`: patch inference after one restore;
- `scripts/wrappers/strict_x4_from_merged_raw.py`: raw candidate  /  exact 4× and intensity 1-NN;
- `scripts/prepare_centerpoint_observed_first_train.py`: N+3N / M+3M;
- `scripts/run_pointrcnn_full_train_stage.py`: PointRCNN adaptation parameters;
- `scripts/run_centerpoint_full_train.py`: CenterPoint adaptation parameters;
- `scripts/run_pointrcnn_checkpoint_split_eval.py`: Visible split, visible LiDAR tree full-val PointRCNN evaluator;
- `scripts/run_patch_causal_centerpoint_eval.py`: CenterPoint evaluator;
- `scripts/run_pugcn_detector_adaptation_full_val_20260908.sh`: Full resumable entry point from generation to 20-arm;
- `scripts/summarize_pugcn_detector_adaptation_full_val_20260908.py`: Only 3769-frame PASS arm is accepted as the final summarizeer.

Full entrance:

```bash
cd /home/ra87racy/projects/baseline_detectors/PointRCNN
bash scripts/run_pugcn_detector_adaptation_full_val_20260908.sh all
```

The stages can also be independent of recovery:`generate_a`, `generate_b`, `prepare`, `pointrcnn`, `centerpoint`, `aggregate`.

Source delivery file:

- `pugcn_full_val_sources.tar.gz`: 151 local source code, CUDA/C++, a byte snapshot of the configuration and split files;
- `source_manifest.json`: relative path, byte and SHA-256 for each file;
- `experiment_source_full.md`: the complete text of the key script of the experiment;
- The snapshot clearly represents the actual version used in the workspace and does not indicate that all documents are consistent with the unmodified version of the author ' s upstream warehouse.

## 12. Current operational status

- 3769-frame Line A patch extraction: All completed;
- Line A PU-GCN inference / strict-4N: A temporary point cloud per 64 frame reasoning, validation and clean-up that can be rebuilt;
- Line B Generation: Waiting for Line A;
- full-val detector matrix: Waiting for the input of two lines to be audited for completeness;
- PointRCNN Single frame end-to-end inference + CPU KITTI AP_R40 + Actual input of audit smoker: PASS.

The final result is only marked after 20/20 arms is all 3769-frame PASS.

I'll see you at runtime. `current_progress.md` with `current_progress.json`, refreshed by the independent progress process every 30 seconds.The generation process is run by 64-frame batch, and only temporary patch/raw/merged files after the strict 4× check have been completed will be cleared;metadata, provenance, final `.bin` and code snapshots can be retained to re-establish the provisional results.
