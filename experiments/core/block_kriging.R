# Kriging for the comparators' variants (the factor setting and the study). Usage: Rscript block_kriging.R <mode> <folder>
# The folder holds training.csv (row, s_x, s_y, value, pred_ins, pred_oof, z_oof), query.csv (row, s_x, s_y) and, for mode 'all',
# params.json. r_ins = value - pred_ins and r_oof = value - pred_oof are formed here.
# Coordinates are projected metres. Every mode writes status.json (one entry per output: "ok" or "failed: <message>").
#   fit  For every value column, an exponential variogram with nugget fitted by automap (autofitVariogram, model = "Exp", automap's
#        default sample variogram and weighted least squares) -> fit_<column>.json: nugget, psill, range (gstat's Exp range parameter).
#        Used on the development splits with n >= 50 to derive the cross-field ("average") variogram shapes.
#   all  For the residual columns r_ins and r_oof: ordinary kriging with (i) the local automap variogram (automap's default families, as
#        kriging.R) and (ii) the fixed cross-field shape of that column (params.json: range in metres, nugget share) with the sill set to
#        the sample variance of the column (divisor n - 1). Each gives the kriging means at the query rows (<name>_mean.csv) and their
#        joint conditional covariance including the nugget and the unknown-mean term (<name>_cov.csv).
#        For the standardised errors z_oof: the correlation C(h)/C(0) between the query rows from (i) the local automap variogram and
#        (ii) the fixed cross-field shape (corr_<name>.csv). Names: rins_loc, rins_fix, roof_loc, roof_fix, z_loc, z_fix.
args <- commandArgs(trailingOnly = TRUE)
stopifnot(length(args) == 2)
mode <- args[1]; folder <- args[2]
if (nzchar(Sys.getenv("SOILGP_RLIB"))) .libPaths(c(Sys.getenv("SOILGP_RLIB"), .libPaths()))
suppressPackageStartupMessages({library(automap); library(gstat); library(sp); library(jsonlite)})

a <- read.csv(file.path(folder, "training.csv"), check.names = FALSE)
b <- read.csv(file.path(folder, "query.csv"), check.names = FALSE)
# Residuals are computed here, as value - prediction, exactly as kriging.R does for block RK: automap's choice among
# near-equal variogram models can flip on last-digit differences, so computing them elsewhere would not reproduce the stored rows.
if ("pred_ins" %in% names(a)) a$r_ins <- a$value - a$pred_ins
if ("pred_oof" %in% names(a)) a$r_oof <- a$value - a$pred_oof
train <- a; coordinates(train) <- ~s_x + s_y
query <- b; coordinates(query) <- ~s_x + s_y
status <- list()

joint_krige <- function(formula, data, newdata, model) {
  # Conditional covariance of ordinary kriging, computed directly; its mean and diagonal are checked against krige() (as kriging.R).
  marginal <- krige(formula, data, newdata, model = model, debug.level = 0)
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
  list(mean = mean, cov = covariance)
}

fixed_model <- function(sill, range, nugget_share) {
  stopifnot(is.finite(sill), sill > 0, is.finite(range), range > 0, nugget_share >= 0, nugget_share < 1)
  vgm(psill = sill * (1 - nugget_share), model = "Exp", range = range, nugget = sill * nugget_share)
}

save_matrix <- function(x, name) write.table(x, file.path(folder, name), sep = ",", row.names = FALSE, col.names = FALSE)

if (mode == "fit") {
  for (col in intersect(c("r_ins", "r_oof", "z_oof"), names(a))) {
    status[[col]] <- tryCatch({
      f <- as.formula(paste0("`", col, "`~1"))
      m <- autofitVariogram(f, train, model = c("Exp"))$var_model
      stopifnot(as.character(m$model[2]) == "Exp")
      write_json(list(nugget = m$psill[1], psill = m$psill[2], range = m$range[2]), file.path(folder, paste0("fit_", col, ".json")),
                 auto_unbox = TRUE, digits = NA)
      "ok"
    }, error = function(e) paste("failed:", conditionMessage(e)))
  }
} else if (mode == "all") {
  p <- fromJSON(file.path(folder, "params.json"))
  parts <- if (is.null(p$parts)) c("loc", "fix") else p$parts      # local and fixed variograms run in separate R processes (block_cases)
  for (col in intersect(c("r_ins", "r_oof"), names(a))) {
    nm <- sub("_", "", col)
    f <- as.formula(paste0(col, "~1"))
    if ("loc" %in% parts) status[[paste0(nm, "_loc")]] <- tryCatch({
      k <- joint_krige(f, train, query, autofitVariogram(f, train)$var_model)
      save_matrix(k$mean, paste0(nm, "_loc_mean.csv")); save_matrix(k$cov, paste0(nm, "_loc_cov.csv")); "ok"
    }, error = function(e) paste("failed:", conditionMessage(e)))
    if ("fix" %in% parts && !is.null(p$shapes[[col]])) {
      status[[paste0(nm, "_fix")]] <- tryCatch({
        k <- joint_krige(f, train, query, fixed_model(var(a[[col]]), p$shapes[[col]]$range, p$shapes[[col]]$nugget_share))
        save_matrix(k$mean, paste0(nm, "_fix_mean.csv")); save_matrix(k$cov, paste0(nm, "_fix_cov.csv")); "ok"
      }, error = function(e) paste("failed:", conditionMessage(e)))
    }
  }
  correlation <- function(model) {
    C <- variogramLine(model, dist_vector = spDists(query, query), covariance = TRUE)
    C0 <- variogramLine(model, dist_vector = matrix(0, 1, 1), covariance = TRUE)[1, 1]
    stopifnot(all(is.finite(C)), C0 > 0)
    C / C0
  }
  if ("loc" %in% parts) status[["z_loc"]] <- tryCatch({
    save_matrix(correlation(autofitVariogram(z_oof ~ 1, train)$var_model), "corr_z_loc.csv"); "ok"
  }, error = function(e) paste("failed:", conditionMessage(e)))
  if ("fix" %in% parts && !is.null(p$shapes[["z_oof"]])) {
    status[["z_fix"]] <- tryCatch({
      save_matrix(correlation(fixed_model(1, p$shapes[["z_oof"]]$range, p$shapes[["z_oof"]]$nugget_share)), "corr_z_fix.csv"); "ok"
    }, error = function(e) paste("failed:", conditionMessage(e)))
  }
} else stop(paste("unknown mode", mode))
write_json(status, file.path(folder, "status.json"), auto_unbox = TRUE, digits = NA)
