# Cloud policy benchmark v1

## Result

The A800 acceptance pack at
`runtime/cloud-policy-cloudsen12-20260920-01` compares the Harness policy
`sentinel-2-scl-cloud-shadow-cirrus-snow-saturation-v1` with CloudSEN12
high-quality manual labels. The 8-window sample contains a concrete
counterexample: at `maximum_cloud_fraction=0.2`, the SCL policy accepts two
windows whose independently labelled invalid fractions exceed 0.2.

This falsifies the narrow assumption that an SCL-policy fraction at or below
0.2 always guarantees at least 80% independently clear pixels. It does not
estimate global Sentinel-2 cloud-mask quality and does not invalidate SCL as a
useful screening signal.

## Truth and policy definitions

CloudSEN12 manual labels use these classes:

| Class | Meaning | Benchmark treatment |
| ---: | --- | --- |
| 0 | clear | usable |
| 1 | thick cloud | invalid |
| 2 | thin cloud | invalid |
| 3 | cloud shadow | invalid |

The Harness policy excludes SCL classes 1, 3, 8, 9, 10 and 11. Confusion counts
treat an invalid pixel as positive. Metrics are computed over the intersection
of valid manual and SCL pixels; a zero denominator produces JSON `null`, not a
fabricated zero or perfect score.

The manual labels are independent of the Sen2Cor SCL output under test: they
were produced by expert photo-interpretation with IRIS initialization and manual
refinement, and IRIS uses s2cloudless rather than Sen2Cor. They are not
statistically independent of the source imagery, and they are not infallible.
The CloudSEN12 paper reports weaker annotator agreement for thin cloud and cloud
shadow, so these labels are a reviewed reference rather than absolute truth.

## Frozen source and data boundary

`config/cloud-policy-benchmark-v1.json` pins:

- dataset DOI `10.57760/sciencedb.06669`;
- paper DOI `10.1038/s41597-022-01878-2`;
- source license `CC BY-NC 4.0` and scope `local-research`;
- `not_redistribution_authorization=true`;
- archive `24__ROI_0549__ROI_0570.tar`, file ID
  `637daff286ce5f243f83c43b`, declared size 1,797,259,897 bytes and ETag
  `637daff2-6b200279`;
- a 134,217,728-byte archive prefix with SHA-256
  `3cf05bf3143973add01c4b155a280f27fa456111987a9b99735a040e67684729`;
- 16 exact TIFF members, their sizes, hashes and reviewed grids;
- thresholds 0.001 and 0.2.

Only the 128 MiB prefix was acquired. The preparer streams the gzip tar, admits
only exact allowlisted regular files and stops after all 16 labels have been
read. It rejects unsafe paths, observed duplicate members, links, size/hash
drift, TIFF profile drift, class-domain drift and unaligned label pairs before
publishing output. It does not require the deliberately truncated gzip stream
to reach archive EOF.

The full 1.8 GB archive and approximately 899 GB CloudSEN12 dataset were not
downloaded. Admitted labels and the source prefix remain ignored runtime data;
neither is licensed for redistribution by this repository configuration.

## Accepted sample and metrics

The pack contains eight `509×509` windows from two ROIs. All 2,072,648 pixels
are jointly valid in this sample.

| Sample | Manual invalid | SCL-policy invalid | Decision at 0.2 |
| --- | ---: | ---: | --- |
| `ROI_0549--20190519T155911_20190519T160743_T18STG` | 0.000000 | 0.000062 | agree accept |
| `ROI_0549--20190718T155911_20190718T160643_T17SQA` | 0.255619 | 0.098869 | **false accept** |
| `ROI_0549--20190928T155029_20190928T155956_T18STG` | 0.947264 | 0.406151 | agree reject |
| `ROI_0549--20200306T155019_20200306T160015_T17SQB` | 0.627364 | 0.328581 | agree reject |
| `ROI_0549--20200707T155819_20200707T161233_T18STF` | 0.822229 | 1.000000 | agree reject |
| `ROI_0550--20190310T023541_20190310T024155_T51RTM` | 1.000000 | 1.000000 | agree reject |
| `ROI_0550--20190504T023559_20190504T024226_T51RTM` | 0.000000 | 0.000000 | agree accept |
| `ROI_0550--20190817T023551_20190817T024442_T51RTM` | 0.229040 | 0.148374 | **false accept** |

Aggregate pixel confusion is:

| TN | FP | FN | TP |
| ---: | ---: | ---: | ---: |
| 1,020,884 | 46,137 | 279,175 | 726,452 |

| Metric | Value |
| --- | ---: |
| precision | 0.940283 |
| recall | 0.722387 |
| intersection over union | 0.690699 |
| specificity | 0.956761 |
| balanced accuracy | 0.839574 |

At threshold 0.2, window decision accuracy is 6/8, with two false accepts and
no false rejects. At threshold 0.001 it is 8/8 for this small sample. The latter
is not evidence of global validity; it only says no counterexample occurred in
the selected eight windows at that stricter threshold.

## Reproduction

Run in the pinned API environment with a fresh ignored runtime output:

```bash
python scripts/prepare_cloud_policy_benchmark.py \
  --archive-prefix /tmp/cloudsen12-range-128m.tar \
  --output runtime/<fresh-cloud-policy-pack> \
  --runtime-root runtime
```

The command verifies the prefix before output, uses the shared storage quota,
writes the 16 private labels with mode 0600 and emits the deterministic
`cloud-policy-validation.json` report. It refuses to overwrite existing output.

## Research boundary

This benchmark establishes a reproducible semantic check that is independent
of the SCL output being tested. Coverage remains only two ROIs and eight windows
from one public archive. Future claims about policy calibration require a
larger, geographically and seasonally stratified sample, confidence intervals
and explicit analysis by thick cloud, thin cloud and shadow class.
