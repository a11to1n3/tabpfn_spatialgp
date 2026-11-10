"""Area of each field (convex hull of its locations, in hectares), its number of locations and the layout of the locations, from the CSV
files of export.R; descriptive only (Table 1 of the paper). The uniformity trials SOIL250 and SOJA98 have grid coordinates, so their
areas are not in hectares.

Layout, from the distance of each location to its nearest neighbour: regular if the coefficient of variation of these distances is at
most 0.2; clustered if the Clark-Evans ratio (their mean over 0.5 * sqrt(hull area / locations), its expectation at random) is below 1;
irregular otherwise.
  python -m experiments.study.field_areas <csv folder> > field_areas.csv"""
import sys
from pathlib import Path

import pandas as pd
from scipy.spatial import ConvexHull, cKDTree

print("field,locations,area_ha,nn_cv,clark_evans,layout")
for p in sorted(Path(sys.argv[1]).glob("*_points.csv")):
    S = pd.read_csv(p)[["x", "y"]].to_numpy(float)
    area = ConvexHull(S).volume; nn = cKDTree(S).query(S, k=2)[0][:, 1]
    cv = nn.std() / nn.mean(); ce = nn.mean() / (0.5 * (area / len(S)) ** 0.5)
    layout = "regular" if cv <= 0.2 else "clustered" if ce < 1 else "irregular"
    print(f"{p.name.replace('_points.csv', '')},{len(S)},{area / 1e4:.2f},{cv:.2f},{ce:.2f},{layout}")
