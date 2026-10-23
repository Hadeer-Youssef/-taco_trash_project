# benchmark_v2 split report
seed: 42

## 1. TACO annotation accounting (v1 silently lost these)

| material | annotations | harvested | crop_failed_or_filtered | too_small | kept% |
|---|---|---|---|---|---|
| cardboard | 251 | 250 | 0 | 1 | 99.6% |
| glass | 254 | 204 | 0 | 50 | 80.3% |
| metal | 550 | 503 | 0 | 47 | 91.5% |
| paper | 246 | 244 | 0 | 2 | 99.2% |
| plastic | 2112 | 2005 | 0 | 107 | 94.9% |
| trash | 1371 | 928 | 0 | 443 | 67.7% |

All 650 dropped annotations were dropped by the absolute `min_size=32px` filter
(`too_small`) — the crop loop itself never failed. The loss is concentrated in
`trash` (443, mostly cigarettes/straw-sized litter) and `glass` (50, broken-glass
shards): exactly the small items a smart bin must catch. A size-relative or
upscale-and-keep policy is a paper ablation, not a silent default.

## 2. Wild photo-grouped split (leakage fix)

wild: 4132 crops / 1494 photos (2.77 crops/photo)

| split | crops | photos | non-recyclable share |
|---|---|---|---|
| train | 3285 | 1192 | 62.8% |
| val | 454 | 151 | 67.2% |
| test | 393 | 151 | 63.1% |

every photo in exactly one split: **OK**

v1 (manifest.csv): 27.6% of photos spanned splits; **80.8% of wild test crops shared a photo with train**. Fixed here.

## 3. Clean domain (TrashNet)

2527 rows; splits copied from manifest.csv (P1 comparability).
Duplicate clusters (md5 or dHash <3): 14
- containing an exact md5 duplicate: 3
- spanning existing splits: 14

## 4. Files

- `manifest_v2.csv` photo/ann_id/bbox stats/ambiguous columns added
- `crop_audit.csv` annotation accounting
- `clean_dups.csv` TrashNet duplicate clusters