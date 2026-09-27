# Original vs Downsampled x4 + Upsampling Quality Metrics

- Generated at: 2026-07-06 22:04:21 UTC

Delta columns = method − Original baseline. Positive delta = worse than original.

| method | source_point_count | gt_reference | cd_total_mean | delta_cd_vs_original | hd_total_mean | delta_hd_vs_original | nuc_mean | delta_nuc_vs_original | p2f_mean | delta_p2f_vs_original | p2f_status | valid_samples |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| Original baseline | 1024 | dense_mesh_surface_10000 | 0.04954787177610021 | 0.0 | 0.11972019562324597 | 0.0 | 1.0619430135945882 | 0.0 | 0.08246926348028343 | 0.0 | exact | 12311 |
| Downsampled x4 + EAR | 1024 | dense_mesh_surface_10000 | 0.0813563966469725 | 0.03180852487087229 | 0.20879327912833842 | 0.08907308350509245 | 0.9577950867942319 | -0.10414792680035634 | 0.08115380757410291 | -0.0013154559061805127 | exact | 12311 |
| Downsampled x4 + PDANS | 1024 | dense_mesh_surface_10000 | 0.0631374116132111 | 0.013589539837110885 | 0.2059271465456305 | 0.08620695092238452 | 1.0899236440044222 | 0.027980630409834006 | 0.07958593782259225 | -0.0028833256576911714 | exact | 12311 |
| Downsampled x4 + PU-Net | 1024 | dense_mesh_surface_10000 | 0.05711532677016791 | 0.0075674549940676974 | 0.1376742752104174 | 0.01795407958717142 | 1.058211690705892 | -0.0037313228886961536 | 0.08006658204365182 | -0.0024026814366316096 | exact | 12311 |
| Downsampled x4 + PU-GCN | 1024 | dense_mesh_surface_10000 | 0.054826907035719186 | 0.0052790352596189735 | 0.16521161166932422 | 0.045491416046078245 | 1.4864687463987774 | 0.42452573280418915 | 0.08377861363436016 | 0.0013093501540767383 | exact | 12311 |
