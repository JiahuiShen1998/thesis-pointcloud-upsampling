# Extended Quality Metrics Comparison

- Generated at: 2026-07-06 22:04:21 UTC

| method | group | line | source_point_count | gt_reference | cd_total_mean | hd_total_mean | nuc_mean | p2f_mean | p2f_status | valid_samples |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| Original baseline | original_baseline_vs_gt | baseline | 1024 | dense_mesh_surface_10000 | 0.04954787177610021 | 0.11972019562324597 | 1.0619430135945882 | 0.08246926348028343 | exact | 12311 |
| Downsampled x4 baseline | downsampled_x4_baseline_vs_gt | lineB_baseline | 256 | dense_mesh_surface_10000 | 0.07712293021476044 | 0.21425127871812702 | 2.0492888877674575 | 0.08249026548917898 | exact | 12311 |
| Downsampled x4 + EAR | downsampled_x4_up_ear_vs_gt | lineB | 1024 | dense_mesh_surface_10000 | 0.0813563966469725 | 0.20879327912833842 | 0.9577950867942319 | 0.08115380757410291 | exact | 12311 |
| Downsampled x4 + PDANS | downsampled_x4_up_pdans_vs_gt | lineB | 1024 | dense_mesh_surface_10000 | 0.0631374116132111 | 0.2059271465456305 | 1.0899236440044222 | 0.07958593782259225 | exact | 12311 |
| Downsampled x4 + PU-Net | downsampled_x4_up_pu_net_vs_gt | lineB | 1024 | dense_mesh_surface_10000 | 0.05711532677016791 | 0.1376742752104174 | 1.058211690705892 | 0.08006658204365182 | exact | 12311 |
| Downsampled x4 + PU-GCN | downsampled_x4_up_pu_gcn_vs_gt | lineB | 1024 | dense_mesh_surface_10000 | 0.054826907035719186 | 0.16521161166932422 | 1.4864687463987774 | 0.08377861363436016 | exact | 12311 |
| Original + EAR | original_up_ear_vs_gt | lineA | 4096 | dense_mesh_surface_10000 | 0.049196694146511157 | 0.11740812648512587 | 0.806502459298751 | 0.08236956640851828 | exact | 12311 |
| Original + PDANS | original_up_pdans_vs_gt | lineA | 4096 | dense_mesh_surface_10000 | 0.04101784375642294 | 0.11543654919397003 | 0.6468133885400085 | 0.08115920089797811 | exact | 12311 |
| Original + PU-Net | original_up_pu_net_vs_gt | lineA | 4096 | dense_mesh_surface_10000 | 0.04452501872658272 | 0.11404317384324736 | 0.5919424725643085 | 0.08004264595539201 | exact | 12311 |
| Original + PU-GCN | original_up_pu_gcn_vs_gt | lineA | 4096 | dense_mesh_surface_10000 | 0.036304399467563635 | 0.09093287558954531 | 0.5513042673706954 | 0.08142620999973965 | exact | 12311 |
