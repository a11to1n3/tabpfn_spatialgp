# Covariates of the SensRes fields Lyduokiai (LT_LYD) and Rumokai (LT_RUM), Lithuania, at their sample points.
# Source: SensRes dataset for downscaling soil maps, Zenodo doi:10.5281/zenodo.14161809 (CC BY 4.0), file SensRes_Lithuania.zip.
# Rasters (all EPSG:3346, 1 m): EMI/<f>/krig_0_025_2.tif (kriged ECa, DUALEM-21H, 2021), Satellite/<f>/Sentinel_Bands_Area_B*.tif
# (Sentinel-2 L2A bare-soil mosaic), Drones/<f>/{red,green,blue}.tif (UAV RGB over bare soil). Values are read at the sample points
# (cell containing the point); nothing is interpolated or transformed.
# Usage: Rscript export_sensres.R <SensRes_Lithuania folder> <output folder>
suppressMessages(library(terra))
args <- commandArgs(trailingOnly = TRUE); root <- args[1]; out <- args[2]
bands <- c("B2", "B3", "B4", "B5", "B6", "B7", "B8", "B8A", "B9", "B11", "B12")
names_out <- c("B02", "B03", "B04", "B05", "B06", "B07", "B08", "B8A", "B09", "B11", "B12")
for (f in c("LT_LYD", "LT_RUM")) {
  obs <- read.delim(file.path(root, "soil_observations", paste0(f, ".txt")))
  pts <- vect(obs, geom = c("X.coordinate", "Y.coordinate"), crs = "EPSG:3346")
  cov <- data.frame(ECa = extract(rast(file.path(root, "EMI", f, "krig_0_025_2.tif")), pts, ID = FALSE)[[1]])
  for (k in seq_along(bands))
    cov[[names_out[k]]] <- extract(rast(file.path(root, "Satellite", f, paste0("Sentinel_Bands_Area_", bands[k], ".tif"))), pts, ID = FALSE)[[1]]
  for (b in c("red", "green", "blue"))
    cov[[paste0("UAV_", b)]] <- extract(rast(file.path(root, "Drones", f, paste0(b, ".tif"))), pts, ID = FALSE)[[1]]
  stopifnot(!anyNA(cov), nrow(cov) == nrow(obs))
  res <- data.frame(ID = obs$ID, x = obs$X.coordinate, y = obs$Y.coordinate, SOC = obs$SOC, clay = obs$clay, pH = obs$pH, cov)
  write.csv(res, file.path(out, paste0(f, "_points.csv")), row.names = FALSE)
  cat(f, ": ", nrow(res), " points, ", ncol(cov), " covariates\n", sep = "")
}
