# The R packages of the study (R 4.2.2): kriging for TabPFN-KpR (experiments/core/kriging.R), block regression kriging and the
# correlogram of Wadoux and Heuvelink (experiments/core/block_kriging.R), and the data export (export_external.R and
# export_sensres.R in experiments/data/).
# Usage: Rscript environment/install_r_packages.R [library folder]; then set SOILGP_RLIB to that folder if it is not the default library.
args <- commandArgs(trailingOnly = TRUE)
lib <- if (length(args)) args[1] else .libPaths()[1]
dir.create(lib, showWarnings = FALSE, recursive = TRUE)
if (!requireNamespace("remotes", quietly = TRUE)) install.packages("remotes", lib = lib, repos = "https://cloud.r-project.org")
versions <- c(sp = "1.6-0", jsonlite = "1.8.4", terra = "1.7-3", sf = "1.0-9", gstat = "2.1-0", reshape = "0.8.10", automap = "1.1-20")
for (p in names(versions)) {
  remotes::install_version(p, version = versions[[p]], lib = lib, repos = "https://cloud.r-project.org", upgrade = "never")
}
.libPaths(c(lib, .libPaths()))
for (p in names(versions)) cat(p, as.character(packageVersion(p, lib.loc = lib)), "\n")
