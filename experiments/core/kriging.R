# Kriging for TabPFN-KpR's features (automap variogram + gstat ordinary kriging, as in the KpR paper's code).
# Usage: Rscript kriging.R kpr <folder>; the folder holds training.csv (row, s_x, s_y, value) and query.csv (row, s_x, s_y).
# Coordinates are projected metres; distances are Euclidean. Leave-one-out kriging at the training rows with the variogram fitted to all
# training rows, and kriging at the query rows -> kpr_training_out.csv, kpr_query_out.csv (row, mean, variance), and status.json.
# Set SOILGP_RLIB to a folder with the R packages (automap, gstat, sp, jsonlite) if they are not installed.
args <- commandArgs(trailingOnly = TRUE)
stopifnot(length(args) == 2)
mode <- args[1]; folder <- args[2]
if (nzchar(Sys.getenv("SOILGP_RLIB"))) .libPaths(c(Sys.getenv("SOILGP_RLIB"), .libPaths()))
suppressPackageStartupMessages({library(automap); library(gstat); library(sp); library(jsonlite)})

a <- read.csv(file.path(folder, "training.csv"), check.names = FALSE)
b <- read.csv(file.path(folder, "query.csv"), check.names = FALSE)
train <- a; coordinates(train) <- ~s_x + s_y
query <- b; coordinates(query) <- ~s_x + s_y
out <- function(name, rows, mean, variance) write.csv(data.frame(row = rows, mean = mean, variance = variance),
                                                       file.path(folder, paste0(name, "_out.csv")), row.names = FALSE)
status <- list()

joint_krige <- function(formula, data, newdata, model) {
  # Conditional covariance of ordinary kriging, computed directly (krige0's fullCovariance rescales prior correlations);
  # its mean and diagonal are checked against krige().
  marginal <- krige(formula, data, newdata, model = model)
  s <- coordinates(data); q <- coordinates(newdata)
  K <- variogramLine(model, dist_vector = spDists(s, s), covariance = TRUE)
  cross <- variogramLine(model, dist_vector = spDists(q, s), covariance = TRUE)
  prior <- variogramLine(model, dist_vector = spDists(q, q), covariance = TRUE)
  factor <- chol(K)
  solve_K <- function(B) backsolve(factor, forwardsolve(t(factor), B))
  inverse_one <- solve_K(rep(1, nrow(s)))
  denominator <- sum(inverse_one)
  Q <- rep(1, nrow(q)) - as.vector(cross %*% inverse_one)
  covariance <- prior - cross %*% solve_K(t(cross)) + outer(Q, Q) / denominator
  covariance <- (covariance + t(covariance)) / 2
  property <- all.vars(formula)[1]
  beta <- sum(solve_K(data@data[[property]])) / denominator
  mean <- as.vector(beta + cross %*% solve_K(data@data[[property]] - beta))
  tolerance <- 1e-6 * max(1, max(abs(marginal$var1.pred)), sum(model$psill))
  stopifnot(max(abs(mean - marginal$var1.pred)) < tolerance, max(abs(diag(covariance) - marginal$var1.var)) < tolerance)
  list(pred = marginal$var1.pred, var = covariance)
}

if (mode == "kpr") {
  model <- autofitVariogram(value ~ 1, train)$var_model
  test <- joint_krige(value ~ 1, train, query, model)
  loo <- data.frame(row = a$row, mean = NA_real_, variance = NA_real_)
  for (i in seq_len(nrow(a))) {
    p <- krige(value ~ 1, train[-i, ], train[i, ], model = model)
    loo$mean[i] <- p$var1.pred; loo$variance[i] <- p$var1.var
  }
  out("kpr_training", loo$row, loo$mean, loo$variance)
  out("kpr_query", b$row, test$pred, diag(test$var))
  status$value <- "ok"
} else stop(paste("unknown mode", mode))
write_json(status, file.path(folder, "status.json"), auto_unbox = TRUE, digits = NA)
