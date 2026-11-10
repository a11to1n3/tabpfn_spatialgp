# Oxford (gstat::oxford) and Meuse (sp::meuse) as CSV for export_data.py. Usage: Rscript export_external.R <folder>
# Covariates as described in the paper's table of development and factor-setting data. Meuse samples without organic matter are dropped.
args <- commandArgs(trailingOnly = TRUE); stopifnot(length(args) == 1)
if (nzchar(Sys.getenv("SOILGP_RLIB"))) .libPaths(c(Sys.getenv("SOILGP_RLIB"), .libPaths()))
suppressPackageStartupMessages({library(gstat); library(sp)})
data(oxford); data(meuse)
ox <- data.frame(x = oxford$XCOORD, y = oxford$YCOORD, ELEV = oxford$ELEV, MAPCLASS = as.integer(oxford$MAPCLASS), VAL1 = oxford$VAL1,
                 CHR1 = oxford$CHR1, OM = oxford$OM1, CLAY = oxford$PCLAY1, PH = oxford$PH1)
me <- meuse[!is.na(meuse$om), ]
mz <- data.frame(x = me$x, y = me$y, elev = me$elev, dist = me$dist, ffreq = as.integer(me$ffreq), soil = as.integer(me$soil), OM = me$om)
write.csv(ox, file.path(args[1], "oxford.csv"), row.names = FALSE)
write.csv(mz, file.path(args[1], "meuse.csv"), row.names = FALSE)
