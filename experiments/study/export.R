# Locations, soil properties and covariates of every field of the study (docs/PROTOCOL.md, Sections 3.1-3.4 and Annex A).
# One reader per group of releases returns a field's released locations and the laboratory determinations of its records; the data rules
# of Section 3.3 and the covariate rule of Section 3.4 are then applied to every field by the same code:
#   rule 1  a location without coordinates, more than 5 km (in the field's coordinate units) from the field's median coordinates, or
#           sharing coordinates with another location is excluded;
#   rule 3  a record (one sample's determinations in one table, or one particle-size analysis) with four or more determinations that all
#           share one value is corrupted: all its values are missing (checked on the values as released);
#   rule 2  a property value outside its plausible range (pH 2-11; organic or total carbon 0-60 % or 0-600 g/kg; organic matter 0-100 %
#           or 0-1000 g/kg; clay 0-100 % or 0-1000 g/kg) is missing;
#   rule 4  repeated values of a property of one sample: identical repeats are used once; for clay from particle-size analyses, the
#           analysis whose five fractions sum exactly to 1000 g/kg (100 %) is used if there is exactly one; otherwise the mean of the
#           distinct values;
#   rule 5  a property missing at more than 10 % of the field's sampled locations (the locations kept by rule 1 that have a released value
#           of at least one of the field's properties) is not used; the field's locations are those with a value of every property used;
#           a field needs at least 30 locations.
# Covariates (Section 3.4), at the field's locations: the supplied covariates that are numeric and known at every location, where a
# raster is read at the cell containing the location and a table of point records (grid nodes or a sensor survey) at its record nearest
# the location, known there if that record lies within 10 m; if no supplied covariate is known at every location, elevation and slope
# from the Copernicus DEM GLO-30 (reprojected to the field's coordinate system at 30 m) and the ten bands of one bare-soil Sentinel-2 L2A
# scene (Earth Search; April, May, October or November 2019-2023; cloud cover at most 10 %; every location's scene classification 4 or 5;
# the lowest median NDVI over the locations; ties to the earliest). A covariate without a value, or constant, at the field's locations is
# dropped; a field left without covariates is not evaluated.
# Writes <out>/<code>_points.csv (x, y, SOC, clay, pH, covariates; a property not used is empty) and <out>/export_log.csv.
# Usage: Rscript experiments/study/export.R <sources root written by fetch.sh> <output folder> [field codes]
suppressMessages({library(terra); library(httr); library(jsonlite); library(readxl)})
args <- commandArgs(trailingOnly = TRUE); root <- normalizePath(args[1]); out <- args[2]
dir.create(out, showWarnings = FALSE, recursive = TRUE)
Sys.setenv(GDAL_DISABLE_READDIR_ON_OPEN = "EMPTY_DIR", CPL_VSIL_CURL_ALLOWED_EXTENSIONS = ".tif", GDAL_HTTP_MAX_RETRY = "5",
           GDAL_HTTP_RETRY_DELAY = "2")
PROPS <- c("SOC", "clay", "pH")
LIMITS <- list(pH = c(2, 11), carbon_pct = c(0, 60), carbon_gkg = c(0, 600), om_pct = c(0, 100), om_gkg = c(0, 1000),
               clay_pct = c(0, 100), clay_gkg = c(0, 1000))
FAR <- 5000; WITHIN <- 10; MIN_N <- 30; MAX_MISSING <- 0.10

# ------------------------------------------------------------------------------------------------------------------------- helpers
num <- function(v) suppressWarnings(as.numeric(trimws(as.character(v))))
num_dec <- function(v) num(gsub(",", ".", as.character(v)))                       # decimal commas (CAREX)
num_thou <- function(v) num(gsub(",", "", as.character(v)))                       # thousands separators (Walnut Creek)
# determinations of wide records: one row per location and determination column; slots maps columns to properties
wide <- function(id, d, cols, slots = c(), parse = num, record = "lab") {
  do.call(rbind, lapply(cols, function(cn) data.frame(id = as.character(id), record = record, det = cn, value = parse(d[[cn]]),
    slot = if (cn %in% names(slots)) slots[[cn]] else "", psa = FALSE, stringsAsFactors = FALSE)))
}
lonlat_to <- function(lon, lat, crs_f) { ok <- is.finite(lon) & is.finite(lat); x <- y <- rep(NA_real_, length(lon))
  if (any(ok)) { p <- crds(project(vect(cbind(lon[ok], lat[ok]), crs = "EPSG:4326"), crs_f)); x[ok] <- p[, 1]; y[ok] <- p[, 2] }
  list(x = x, y = y) }
nearest <- function(xy, sxy, vals) {                                              # point records: nearest record, known within 10 m
  k <- vapply(seq_len(nrow(xy)), function(i) which.min((sxy[, 1] - xy[i, 1])^2 + (sxy[, 2] - xy[i, 2])^2), integer(1))
  dd <- sqrt((sxy[k, 1] - xy[, 1])^2 + (sxy[k, 2] - xy[, 2])^2)
  lapply(vals, function(v) ifelse(dd <= WITHIN, num(v)[k], NA_real_))
}

# ------------------------------------------------------------------------------------------------------------------------- public covariates
bands <- c(B02 = "blue", B03 = "green", B04 = "red", B05 = "rededge1", B06 = "rededge2", B07 = "rededge3", B08 = "nir",
           B8A = "nir08", B11 = "swir16", B12 = "swir22")
stac_items <- function(bb) {                                       # Earth Search items intersecting the lon/lat bounding box, 2019-2023
  body <- list(collections = list("sentinel-2-l2a"), bbox = unname(bb), datetime = "2019-01-01T00:00:00Z/2023-12-31T23:59:59Z",
               query = list(`eo:cloud_cover` = list(lte = 10)), limit = 100)
  items <- list(); url <- "https://earth-search.aws.element84.com/v1/search"
  repeat {
    r <- POST(url, body = toJSON(body, auto_unbox = TRUE, digits = NA), content_type_json(), timeout(120)); stop_for_status(r)
    j <- fromJSON(content(r, "text", encoding = "UTF-8"), simplifyVector = FALSE); items <- c(items, j$features)
    nxt <- Filter(function(l) l$rel == "next", j$links)
    if (length(nxt) == 0) break
    body <- modifyList(body, nxt[[1]]$body)
  }
  items
}
read_at <- function(href, pts) {                                   # values of a COG at the cells containing the points
  r <- rast(paste0("/vsicurl/", href)); extract(r, project(pts, crs(r)), ID = FALSE, method = "simple")[[1]]
}
s2_scene <- function(ll) {                                         # the bare-soil scene with the lowest median NDVI
  e <- as.vector(ext(ll)); items <- stac_items(c(e[1], e[3], e[2], e[4]))
  ok <- Filter(function(it) {
    cc <- it$properties[["eo:cloud_cover"]]; mo <- as.integer(substr(it$properties$datetime, 6, 7))
    !is.null(cc) && cc <= 10 && mo %in% c(4, 5, 10, 11)
  }, items)
  ok <- ok[order(sapply(ok, function(it) it$properties$datetime))]
  best <- NULL; best_ndvi <- Inf; n_elig <- 0
  for (it in ok) {
    scl <- tryCatch(read_at(it$assets$scl$href, ll), error = function(err) NA)
    if (any(!is.finite(scl)) || !all(scl %in% c(4, 5))) next
    red <- tryCatch(read_at(it$assets$red$href, ll), error = function(err) NA)
    nir <- tryCatch(read_at(it$assets$nir$href, ll), error = function(err) NA)
    if (any(!is.finite(red)) || any(!is.finite(nir))) next
    n_elig <- n_elig + 1; nd <- median((nir - red) / (nir + red))
    if (is.finite(nd) && nd < best_ndvi) { best <- it; best_ndvi <- nd }
  }
  cov <- list()
  if (!is.null(best)) for (b in names(bands)) cov[[b]] <- read_at(best$assets[[bands[[b]]]]$href, ll)
  list(cov = cov, info = sprintf("%d candidate scenes, %d eligible; scene %s (median NDVI %s)", length(ok), n_elig,
                                 if (is.null(best)) "none" else best$id, if (is.null(best)) "-" else round(best_ndvi, 3)))
}
dem_cov <- function(ll, crs_f) {                                   # Copernicus DEM GLO-30: elevation and slope at the locations
  e <- as.vector(ext(ll)); tiles <- list()
  for (la in floor(e[3]):floor(e[4])) for (lo in floor(e[1]):floor(e[2])) {
    nm <- sprintf("Copernicus_DSM_COG_10_%s%02d_00_%s%03d_00_DEM", if (la >= 0) "N" else "S", abs(la), if (lo >= 0) "E" else "W", abs(lo))
    tiles[[length(tiles) + 1]] <- rast(sprintf("/vsicurl/https://copernicus-dem-30m.s3.amazonaws.com/%s/%s.tif", nm, nm))
  }
  d <- if (length(tiles) == 1) tiles[[1]] else do.call(merge, tiles)
  pts <- project(ll, crs_f); b <- ext(pts); b <- ext(b[1] - 300, b[2] + 300, b[3] - 300, b[4] + 300)
  win <- crop(d, ext(project(as.polygons(b, crs = crs_f), crs(d))) * 1.2)
  tmpl <- rast(b, resolution = 30, crs = crs_f); dm <- project(win, tmpl, method = "bilinear")
  list(elevation = extract(dm, pts, ID = FALSE, method = "simple")[[1]],
       slope = extract(terrain(dm, "slope", neighbors = 8, unit = "degrees"), pts, ID = FALSE, method = "simple")[[1]])
}
usable <- function(cov) Filter(function(v) is.numeric(v) && all(is.finite(v)) && length(unique(v)) > 1, cov)

# ------------------------------------------------------------------------------------------------------------------------- readers
# Each reader returns list(loc = data.frame(id, x, y), crs, det = data.frame(id, record, det, value, slot, psa), kind = c(SOC = "carbon"
# or "om"), unit = c(SOC = "pct" or "gkg", clay = ...), sup = function(xy) -> list of supplied covariates (or NULL), sup_info, combine =
# optional function(values by id x slot) -> SOC, clay, pH by id).
trial <- function(code) {
  e <- new.env(); load(file.path(root, "geor/x/geoR/data", paste0(tolower(code), ".rda")), envir = e); d <- get(tolower(code), e)
  id <- as.character(seq_len(nrow(d)))
  if (code == "SOIL250") {
    cols <- c("AGrossa", "Silte", "Argila", "pHAgua", "pHKCl", "Ca", "Mg", "K", "Al", "H", "C", "N", "CTC", "S", "V", "M", "NC", "CEC", "CN")
    list(loc = data.frame(id = id, x = num(d$Linha), y = num(d$Coluna)), crs = NA,
         det = wide(id, d, cols, c(C = "SOC", Argila = "clay", pHAgua = "pH")), kind = c(SOC = "carbon"), unit = c(SOC = "pct", clay = "pct"),
         sup = function(xy) list(Cota = num(d$Cota)), sup_info = "supplied: Cota")
  } else {
    list(loc = data.frame(id = id, x = num(d$X), y = num(d$Y)), crs = NA,
         det = wide(id, d, c("P", "PH", "K", "MO", "SB"), c(MO = "SOC", PH = "pH")), kind = c(SOC = "om"), unit = c(SOC = "gkg"),
         sup = function(xy) list(PROD = num(d$PROD), iCone = num(d$iCone)), sup_info = "supplied: PROD, iCone")
  }
}

SR <- file.path(root, "sensres"); NI <- file.path(SR, "Northern_Ireland", "SensRes_Northern_Ireland")
band_name <- function(f) {                                        # B2, B02_10m, Sentinel_Bands_Area_B8A -> B02, B02, B8A
  b <- regmatches(f, regexpr("B[0-9]+A?", f)); n <- sub("A$", "", sub("^B", "", b))
  paste0("B", sprintf("%02d", as.integer(n)), ifelse(grepl("A$", b), "A", ""))
}
cov_name <- function(path) {
  f <- basename(path); d <- basename(dirname(dirname(path)))
  if (grepl("EMI", d)) return("ECa")
  if (grepl("Gamma", d)) return(paste0("Gamma_", sub("\\.tif$", "", f)))
  if (grepl("Satellite", d)) return(band_name(f))
  if (f %in% c("red.tif", "green.tif", "blue.tif")) return(paste0("UAV_", sub("\\.tif$", "", f)))
  sub("\\.tif$", "", f)
}
sensres <- function(code) {
  dk <- c(DK_EST = "Estrup", DK_FAA = "Faardrup", DK_JYN = "Jyndevad", DK_SIL = "Silstrup", DK_VIN = "Vindum")
  ni <- c(NI_BRA = "Brecart_A", NI_MAR = "Maryland")
  tr <- c(TR_CIH = "cihanbeyli", TR_KAD = "kadinhani", TR_KAR = "karapinar", TR_KAT = "karatay")
  if (code %in% names(dk)) {
    s <- dk[[code]]; obs <- file.path(SR, "Denmark/soil_observations", paste0(s, "_coords_tex.txt")); xyc <- c("X.coordinate", "Y.coordinate")
    idc <- "ID"; cols <- c("clay", "silt", "sand_f", "sand_c", "humus"); slots <- c(humus = "SOC", clay = "clay"); kind <- "om"
    dirs <- file.path(SR, "Denmark", c("covariates_EMI", "Covariates_Gamma_DK_2", "covariates_Satellite", "covariates_Drone"), s)
  } else if (code %in% names(ni)) {
    s <- ni[[code]]; obs <- file.path(NI, "soil_observations", paste0(s, "_coords_tex.txt")); xyc <- c("POINT_X", "POINT_Y"); idc <- "ID1"
    cols <- c("SoilpH", "SoilP", "SoilK", "SoilMg", "SoilS", "SoilLOI", "LOI", "SoilTN", "SoilTC", "Sand", "Silt", "Clay")
    slots <- c(SoilTC = "SOC", Clay = "clay", SoilpH = "pH"); kind <- "carbon"
    dirs <- file.path(NI, c("EMI", "Satellite", "Drone"), s)
  } else {
    s <- tr[[code]]; obs <- file.path(SR, "Turkey/soil_observations", paste0(s, ".txt.txt")); xyc <- c("X.coordinate", "Y.coordinate")
    idc <- "ID"; cols <- c("clay", "silt", "sand", "saturation", "ph", "EC", "CaCO3", "soc"); slots <- c(soc = "SOC", clay = "clay", ph = "pH")
    kind <- "carbon"; dirs <- file.path(SR, "Turkey", c("Satellite", "Drones"), s)
  }
  d <- read.delim(obs, check.names = TRUE, fileEncoding = "latin1"); id <- as.character(d[[idc]]); cols <- intersect(cols, names(d))
  dirs <- dirs[file.exists(dirs)]; tifs <- unlist(lapply(dirs, function(x) sort(list.files(x, pattern = "\\.tif$", full.names = TRUE))))
  crs_f <- crs(rast(tifs[1]))
  sup <- function(xy) { pts <- vect(xy, crs = crs_f); cov <- list()
    for (p in tifs) cov[[cov_name(p)]] <- extract(rast(p), pts, ID = FALSE)[[1]]
    cov }
  list(loc = data.frame(id = id, x = num(d[[xyc[1]]]), y = num(d[[xyc[2]]])), crs = crs_f, det = wide(id, d, cols, slots),
       kind = c(SOC = kind), unit = c(SOC = "pct", clay = "pct"), sup = sup, sup_info = "supplied: sensor rasters")
}

midwest <- function(code) {
  site <- c(PO_NE = "NE", PO_RS = "IL-RS", PO_PT = "IL-PT", PO_DG = "IL-DG", PO_MC = "IL-MC", PO_BR = "IL-BR", PO_RD = "IL-RD",
            PO_RT = "IL-RT")[[code]]
  crs_f <- paste0("EPSG:", c(PO_NE = 32614, PO_BR = 32615)[code]); if (crs_f == "EPSG:NA") crs_f <- "EPSG:32616"
  l <- read.csv(file.path(root, "midwest/locations.csv"), colClasses = c(location_id = "character")); l <- l[l$site == site, ]
  m <- read.csv(file.path(root, "midwest/measurements.csv"), colClasses = c(location_id = "character"))
  m <- m[m$site == site & m$sample_depth_max <= 30, ]
  rec <- paste0(m$sample_depth_min, "-", m$sample_depth_max)      # one record per depth segment; SOC concentration and bulk density
  det <- rbind(data.frame(id = m$location_id, record = rec, det = "SOCc", value = num(m$SOCc), slot = paste0("SOC@", rec), psa = FALSE),
               data.frame(id = m$location_id, record = rec, det = "BD", value = num(m$BD), slot = "", psa = FALSE))
  combine <- function(v, released) {                              # thickness-weighted mean over segments covering exactly 0-30 cm
    out <- matrix(NA_real_, nrow(v), 3, dimnames = list(rownames(v), PROPS))
    sc <- grep("^SOC@", colnames(v), value = TRUE); lo <- as.numeric(sub("^SOC@(.*)-.*$", "\\1", sc)); hi <- as.numeric(sub("^.*-", "", sc))
    for (i in seq_len(nrow(v))) {
      k <- which(released[i, sc]); k <- k[order(lo[k])]
      if (!length(k) || lo[k[1]] != 0 || max(hi[k]) != 30 || any(head(hi[k], -1) != tail(lo[k], -1)) || anyNA(v[i, sc[k]])) next
      out[i, "SOC"] <- sum(v[i, sc[k]] * (hi[k] - lo[k])) / 30
    }
    out
  }
  xy <- lonlat_to(num(l$X), num(l$Y), crs_f)
  list(loc = data.frame(id = l$location_id, x = xy$x, y = xy$y), crs = crs_f, det = det, kind = c(SOC = "carbon"), unit = c(SOC = "pct"),
       sup = function(xy) NULL, sup_info = "none supplied", combine = combine)
}

new_fields <- function(code) {
  NX <- file.path(root, "newfields")
  if (code == "NX_ON") {
    d <- read.csv(file.path(NX, "ontario", "2. SOC_Database.csv"), check.names = FALSE); id <- as.character(d$SampleID)
    xy <- lonlat_to(num(d[["X (DD)"]]), num(d[["Y (DD)"]]), "EPSG:32617")
    sc <- setdiff(names(d), c("SampleID", "SOC (%)", "X (DD)", "Y (DD)"))
    list(loc = data.frame(id = id, x = xy$x, y = xy$y), crs = "EPSG:32617", det = wide(id, d, "SOC (%)", c(`SOC (%)` = "SOC")),
         kind = c(SOC = "carbon"), unit = c(SOC = "pct"), sup = function(xy) lapply(d[sc], num), sup_info = "supplied: every covariate column")
  } else if (startsWith(code, "NX_CZ")) {
    d <- as.data.frame(read_excel(file.path(NX, "czech", "STEROPES_CZU_Collected_samples.xlsx"), sheet = 1, .name_repair = "minimal"))[, 1:8]
    d <- d[substr(d$Sample_Code, 1, 1) == substr(code, 6, 6), ]; id <- as.character(d$Sample_Code)
    list(loc = data.frame(id = id, x = num(d$Longtitude), y = num(d$Latitude)), crs = "EPSG:32633",
         det = wide(id, d, c("SOC", "Clay", "Silt", "Sand"), c(SOC = "SOC", Clay = "clay")), kind = c(SOC = "carbon"),
         unit = c(SOC = "pct", clay = "pct"), sup = function(xy) NULL, sup_info = "none supplied")
  } else if (startsWith(code, "NX_IT")) {
    d <- as.data.frame(vect(file.path(NX, "italy", sprintf("Farm_%02d_SOC.shp", as.integer(substr(code, 6, 6))))))
    id <- as.character(d$ID_Sample); xy <- lonlat_to(num(d$Lon), num(d$Lat), "EPSG:32632")
    list(loc = data.frame(id = id, x = xy$x, y = xy$y), crs = "EPSG:32632", det = wide(id, d, "CO", c(CO = "SOC")), kind = c(SOC = "carbon"),
         unit = c(SOC = "gkg"), sup = function(xy) NULL, sup_info = "none supplied")
  } else if (code == "NX_CAF") {
    td <- tempfile(); untar(file.path(NX, "cookfarm", "GSIF_0.5-5.1.tar.gz"), files = "GSIF/data/cookfarm.rda", exdir = td)
    e <- new.env(); load(file.path(td, "GSIF/data/cookfarm.rda"), envir = e); p <- e$cookfarm$profiles; g <- e$cookfarm$grids
    pr <- unique(p[, c("SOURCEID", "Easting", "Northing")]); top <- p[p$UHDICM == 0, ]
    rec <- paste0("h0_", ave(seq_len(nrow(top)), top$SOURCEID, FUN = seq_along))   # records of the horizon starting at 0 cm
    det <- rbind(wide(top$SOURCEID, top, "PHIHOX", c(PHIHOX = "pH")), wide(top$SOURCEID, top, "BLD")); det$record <- rep(rec, 2)
    vars <- c("DEM", "TWI", "NDRE.M", "NDRE.Sd", "Cook_fall_ECa", "Cook_spr_ECa")
    list(loc = data.frame(id = as.character(pr$SOURCEID), x = num(pr$Easting), y = num(pr$Northing)), crs = "EPSG:26911", det = det,
         kind = c(), unit = c(), sup = function(xy) nearest(xy, cbind(g$x, g$y), g[vars]), sup_info = "supplied: 10 m grid")
  } else {
    so <- read.csv(file.path(NX, "swiss", "sample_origin.csv"), comment.char = "#")
    sp <- read.csv(file.path(NX, "swiss", "soil_properties.csv"), comment.char = "#")
    d <- merge(so, sp, by = "sample_ID"); d <- d[d$field_ID == substr(code, 6, 7), ]; id <- as.character(d$sample_ID)
    xy <- lonlat_to(num(d$long), num(d$lat), "EPSG:32632")
    list(loc = data.frame(id = id, x = xy$x, y = xy$y), crs = "EPSG:32632",
         det = wide(id, d, setdiff(names(sp), "sample_ID"), c(C_org = "SOC", clay = "clay", pH_H2O = "pH")), kind = c(SOC = "carbon"),
         unit = c(SOC = "gkg", clay = "gkg"), sup = function(xy) NULL, sup_info = "none supplied")
  }
}

read_carex <- function(p) {                                        # tab- or semicolon-separated, Latin-1 or UTF-8 with BOM
  enc <- if (identical(readBin(p, "raw", 3), as.raw(c(0xef, 0xbb, 0xbf)))) "UTF-8-BOM" else "latin1"
  l1 <- readLines(p, n = 1, encoding = enc, warn = FALSE)
  d <- read.table(p, sep = if (grepl(";", l1)) ";" else "\t", header = TRUE, quote = "\"", fileEncoding = enc, stringsAsFactors = FALSE,
                  comment.char = "", check.names = FALSE)
  names(d) <- trimws(names(d)); d[, names(d) != "" & !startsWith(names(d), "V1"), drop = FALSE]
}
RES <- list(auzeville = "resistivite.raw", epoisses = "x_res/fichiers_finaux/nov_2011.shp", crouel = "resistivite.raw",
            ploudaniel = c("x_res/2012_inra_L93_net_grid.csv", "x_res/2016_inra_L93_net_grid_corrected.csv"))
carex_res <- function(site) {                                      # resistivity records: coordinates and channels 1-3
  parts <- lapply(RES[[site]], function(f) {
    p <- file.path(root, "furtherfields", paste0("carex_", site), f)
    if (endsWith(f, ".shp")) { v <- vect(p); g <- as.data.frame(v); xy <- crds(v); g <- data.frame(x = xy[, 1], y = xy[, 2], g[, c("voie_1", "voie_2", "voie_3")]) }
    else if (site == "auzeville") { g <- read.table(p, header = TRUE, check.names = FALSE); names(g) <- c("x", "y", "voie_1", "voie_2", "voie_3") }
    else { g <- read.table(p, sep = ";", header = TRUE, dec = ",", fileEncoding = "UTF-8-BOM"); names(g)[1:2] <- c("x", "y") }
    data.frame(lapply(g[, c("x", "y", "voie_1", "voie_2", "voie_3")], num_dec))
  })
  do.call(rbind, parts)
}
further_fields <- function(code) {
  D <- file.path(root, "furtherfields")
  CAREX <- c(DR_AUZ = "auzeville", DR_EPO = "epoisses", DR_PLO = "ploudaniel", DR_CRO = "crouel")
  if (code %in% names(CAREX)) {
    site <- CAREX[[code]]; dir <- file.path(D, paste0("carex_", site))
    pos <- read_carex(file.path(dir, "positions.raw")); id <- as.character(pos$id_profil)
    ch <- read_carex(file.path(dir, "analyses_chimiques.raw")); ch <- ch[trimws(ch$no_horizon) == "1", ]
    slot <- ifelse(ch$determination == "ph_eau", "pH", ifelse(ch$determination == "c_org" & site != "epoisses", "SOC", ""))
    det <- data.frame(id = as.character(ch$id_profil), record = "chem", det = ch$determination, value = num_dec(ch$valeur), slot = slot,
                      psa = FALSE)
    gr <- read_carex(file.path(dir, "granulometrie_5_fractions.raw")); gr <- gr[trimws(gr$no_horizon) == "1", ]
    k <- ave(as.integer(num_dec(gr$seuil_inferieur) == 0), gr$id_profil, FUN = cumsum)   # analysis number: a new one starts at 0 um
    det <- rbind(det, data.frame(id = as.character(gr$id_profil), record = paste0("psa", k), det = gr$classe_granulo,
                                 value = num_dec(gr$teneur_g_kg), slot = ifelse(startsWith(gr$classe_granulo, "Argile"), "clay", ""), psa = TRUE))
    g <- NULL
    sup <- function(xy) { if (is.null(g)) g <<- carex_res(site)
      nearest(xy, cbind(g$x, g$y), list(res_v1 = g$voie_1, res_v2 = g$voie_2, res_v3 = g$voie_3)) }
    list(loc = data.frame(id = id, x = num_dec(pos$x_l93), y = num_dec(pos$y_l93)), crs = "EPSG:2154", det = det, kind = c(SOC = "carbon"),
         unit = c(SOC = "gkg", clay = "gkg"), sup = sup, sup_info = "supplied: electrical resistivity, channels 1-3")
  } else if (code %in% c("DR_F013", "DR_F033", "DR_F111")) {
    fld <- c(DR_F013 = "F013B4", DR_F033 = "F033", DR_F111 = "F111")[[code]]
    v <- as.data.frame(vect(file.path(D, "maricopa", paste0(fld, "_SoilPhysicalAnalysis_DJH.shp")))); id <- as.character(v$Core)
    g <- fromJSON(file.path(D, "maricopa", paste0(fld, "_SoilPhysicalAnalysis_DJH.geojson")), simplifyVector = FALSE)$features
    at15 <- function(ft, nm) { c <- ft$properties[[nm]]; w <- Filter(function(z) identical(as.integer(z$depth), 15L), c)
      if (length(w) && !is.null(w[[1]]$value)) num(w[[1]]$value) else NA_real_ }
    cols <- c("SLSND_M", "SLSLT_M", "SLCLY_M", "SLWP", "SLWP2", "SLFC1")       # numeric determinations of the 0-30 cm layer
    lab <- vapply(g, function(ft) as.character(ft$properties$spa_label), "")
    stopifnot(!anyDuplicated(lab))
    w <- as.data.frame(lapply(setNames(cols, cols), function(nm) vapply(g, at15, 0, nm = nm)))
    sup <- if (fld == "F033") function(xy) {
      out <- list()
      for (o in c("H", "V")) { m <- read.csv(file.path(D, "maricopa", sprintf("F033_EM38%s_2009d079.csv", o)))
        out[[paste0("EM38", o)]] <- nearest(xy, cbind(m$UTMX, m$UTMY), list(m$Mean))[[1]] }
      out } else function(xy) NULL
    list(loc = data.frame(id = id, x = num(v$UTMX), y = num(v$UTMY)), crs = "EPSG:32612",
         det = wide(lab, w, cols, c(SLCLY_M = "clay")), kind = c(), unit = c(clay = "pct"), sup = sup,
         sup_info = if (fld == "F033") "supplied: EM38 horizontal and vertical, 20 March 2009" else "none with defined columns")
  } else if (code == "DR_WAL") {
    s <- read.csv(file.path(D, "walnutcreek", "Soil_properties_0.csv"), check.names = FALSE); names(s) <- trimws(names(s))
    t <- read.csv(file.path(D, "walnutcreek", "Topograhic_metrics_0.csv"), check.names = FALSE); names(t) <- trimws(names(t))
    id <- as.character(s$idendifica); k <- match(s$idendifica, t$idendifica)
    cols <- c("BD", "Z_SAND", "ZSILT", "ZCLAY", "mSOC", "_13C", "mC3", "mC4", "Cs137", "SR")
    list(loc = data.frame(id = id, x = num_thou(s$x), y = num_thou(s$y)), crs = "EPSG:26915",
         det = wide(id, s, cols, c(ZCLAY = "clay"), parse = num_thou), kind = c(), unit = c(clay = "pct"),
         sup = function(xy) lapply(t[k, setdiff(names(t), c("idendifica", "x", "y"))], num_thou), sup_info = "supplied: terrain metrics")
  } else if (code %in% c("DR_HEN", "DR_NWY")) {
    e <- read.csv(file.path(D, "eidc_sheep", "x", "data", "extensive_intensive_spatial.csv"), check.names = FALSE, fileEncoding = "UTF-8-BOM")
    names(e) <- trimws(names(e)); e <- e[e$Site == (if (code == "DR_HEN") "Extensive" else "Intensive"), ]; id <- as.character(e$Sample_ID)
    cols <- c("Bulk_density", "WFPS", "Soil_moisture", "Organic_Matter", "Soil_pH", "Soil_NO3N", "Soil_NH4N", "Total_C", "Total_N", "CtoN")
    sc <- c("Elevation", "Slope", "CTI", "Aspect", "Veg_type/Org_Spread")
    list(loc = data.frame(id = id, x = num(e$Easting), y = num(e$Northing)), crs = "EPSG:27700",
         det = wide(id, e, cols, c(Organic_Matter = "SOC", Soil_pH = "pH")), kind = c(SOC = "om"), unit = c(SOC = "pct"),
         sup = function(xy) lapply(e[sc], num), sup_info = "supplied: site characteristics")
  } else {
    d <- as.data.frame(read_excel(file.path(D, "caroline", "x", "LCB_Data", "LCB_Legacy P_Soil_Sediment.xlsx"), sheet = "Data Summary"),
                       check.names = FALSE)
    d <- d[which(d$`Sample Type` == "Soil" & d$Depth_cm == "0_5" & num(d$Fieldboundary) == 1), ]; id <- as.character(d$Object_ID)
    xy <- lonlat_to(num(d$Location_lon), num(d$Location_lat), "EPSG:32618")
    cols <- names(d)[match("pH", names(d)):match("Sand_pct", names(d))]
    tm <- c("Ele", "CA", "FA", "FPL", "GCur", "LS", "PCur", "PlCu", "POP", "SPI", "Slope", "TWI", "UpSl", "RePC1", "RePC2")
    list(loc = data.frame(id = id, x = xy$x, y = xy$y), crs = "EPSG:32618", det = wide(id, d, cols, c(Clay_pct = "clay", pH = "pH")),
         kind = c(), unit = c(clay = "pct"), sup = function(xy) lapply(d[tm], num), sup_info = "supplied: terrain metrics")
  }
}

reader <- function(code) {
  if (code %in% c("SOIL250", "SOJA98")) trial(code) else if (grepl("^(DK|NI|TR)_", code)) sensres(code)
  else if (startsWith(code, "PO_")) midwest(code) else if (startsWith(code, "NX_")) new_fields(code) else further_fields(code)
}
GROUP <- function(code) if (code %in% c("SOIL250", "SOJA98")) "Uniformity trials" else if (grepl("^(DK|NI|TR)_", code)) "SensRes" else
  if (startsWith(code, "PO_")) "Midwest" else if (startsWith(code, "NX_")) "New fields" else "Further fields"

# ------------------------------------------------------------------------------------------------------------------------- the rules
limits <- function(fd, base) {
  if (base == "pH") return(LIMITS$pH)
  if (base == "SOC") return(LIMITS[[paste0(if (fd$kind[["SOC"]] == "om") "om" else "carbon", "_", fd$unit[["SOC"]])]])
  LIMITS[[paste0("clay_", fd$unit[["clay"]])]]
}
apply_rules <- function(code, fd) {
  L <- fd$loc; L$id <- as.character(L$id); stopifnot(!anyDuplicated(L$id))
  log <- list(code = code, group = GROUP(code), released = nrow(L))
  # rule 1
  has <- is.finite(L$x) & is.finite(L$y)
  far <- has & sqrt((L$x - median(L$x[has]))^2 + (L$y - median(L$y[has]))^2) > FAR
  key <- paste(L$x, L$y); shared <- has & (duplicated(key) | duplicated(key, fromLast = TRUE))
  keep1 <- has & !far & !shared
  log[c("no_coordinates", "far", "shared")] <- list(sum(!has), sum(far), sum(shared))
  L <- L[keep1, ]; d <- fd$det; d$id <- as.character(d$id); d <- d[d$id %in% L$id, ]
  released <- d[d$slot != "" & !is.na(d$value), c("id", "slot")]
  # rule 3 (on the values as released)
  rk <- paste(d$id, d$record, sep = "\r")
  bad <- tapply(d$value, rk, function(v) { v <- v[!is.na(v)]; length(v) >= 4 && length(unique(v)) == 1 })
  bad <- names(bad)[bad]; d$value[rk %in% bad] <- NA
  log$corrupted <- length(bad)
  # rule 2
  base <- sub("@.*", "", d$slot); impl <- rep(FALSE, nrow(d))
  for (b in intersect(unique(base[d$slot != ""]), PROPS)) {
    i <- which(base == b & !is.na(d$value)); lim <- limits(fd, b); o <- d$value[i] < lim[1] | d$value[i] > lim[2]
    impl[i[o]] <- TRUE; log[[paste0("implausible_", b)]] <- sum(o)
  }
  d$value[impl] <- NA
  # rule 4
  slots <- sort(unique(d$slot[d$slot != ""])); V <- matrix(NA_real_, nrow(L), length(slots), dimnames = list(L$id, slots))
  R <- V; R[] <- FALSE; R <- matrix(as.logical(R), nrow(L), dimnames = dimnames(V))
  for (j in seq_len(nrow(released))) R[released$id[j], released$slot[j]] <- TRUE
  repeats <- 0; chosen <- 0
  for (s in slots) {
    q <- d[d$slot == s, ]
    for (g in split(q, q$id)) {
      v <- g$value
      if (nrow(g) > 1) {
        repeats <- repeats + 1
        if (all(g$psa) && length(unique(g$record)) == nrow(g)) {
          tot <- if (s == "clay" && fd$unit[["clay"]] == "pct") 100 else 1000
          sums <- vapply(g$record, function(r) { f <- d$value[d$id == g$id[1] & d$record == r & d$psa]
            if (length(f) == 5 && !anyNA(f)) sum(f) else NA_real_ }, 0)
          hit <- which(!is.na(sums) & sums == tot)
          if (length(hit) == 1) { V[g$id[1], s] <- v[hit]; chosen <- chosen + 1; next }
        }
      }
      u <- unique(v[!is.na(v)]); V[g$id[1], s] <- if (length(u)) mean(u) else NA_real_
    }
  }
  log$repeated <- repeats; log$repeat_by_sum <- chosen
  Y <- if (is.null(fd$combine)) { o <- matrix(NA_real_, nrow(L), 3, dimnames = list(L$id, PROPS)); for (p in intersect(PROPS, slots)) o[, p] <- V[, p]; o }
       else fd$combine(V, R)
  # rule 5
  sampled <- rowSums(R) > 0; L <- L[sampled, ]; Y <- Y[sampled, , drop = FALSE]
  cand <- intersect(PROPS, unique(sub("@.*", "", slots))); miss <- colMeans(is.na(Y[, cand, drop = FALSE]))
  used <- cand[miss <= MAX_MISSING]
  log$sampled <- nrow(L); log$missing <- paste(sprintf("%s %.3f", cand, miss), collapse = "; ")
  log$properties <- paste(used, collapse = " "); log$not_used <- paste(setdiff(cand, used), collapse = " ")
  ok <- if (length(used)) rowSums(is.na(Y[, used, drop = FALSE])) == 0 else rep(FALSE, nrow(L))
  L <- L[ok, ]; Y <- Y[ok, , drop = FALSE]; Y[, setdiff(PROPS, used)] <- NA
  log$N <- nrow(L)
  list(L = L, Y = Y, used = used, log = log)
}

covariates <- function(fd, L) {
  xy <- cbind(L$x, L$y); sup <- fd$sup(xy); s <- if (is.null(sup)) list() else usable(lapply(sup, function(v) if (is.numeric(v)) v else num(v)))
  if (length(s)) return(list(cov = s, info = fd$sup_info))
  stopifnot(!is.na(fd$crs))
  ll <- project(vect(xy, crs = fd$crs), "EPSG:4326"); dm <- dem_cov(ll, fd$crs); sc <- s2_scene(ll)
  list(cov = usable(c(dm, sc$cov)), info = paste0("public (", fd$sup_info, "); ", sc$info))
}

# ------------------------------------------------------------------------------------------------------------------------- main
CODES <- c("SOIL250", "SOJA98", "DK_EST", "DK_FAA", "DK_JYN", "DK_SIL", "DK_VIN", "NI_BRA", "NI_MAR", "TR_CIH", "TR_KAD", "TR_KAR",
           "TR_KAT", "PO_NE", "PO_RS", "PO_PT", "PO_DG", "PO_MC", "PO_BR", "PO_RD", "PO_RT", "NX_ON", "NX_CZK", "NX_CZN", "NX_CZP", "NX_CZU",
           "NX_IT1", "NX_IT2", "NX_IT3", "NX_CAF", "NX_CHHB", "NX_CHZO", "DR_AUZ", "DR_EPO", "DR_PLO", "DR_CRO", "DR_F013", "DR_F033",
           "DR_F111", "DR_WAL", "DR_HEN", "DR_NWY", "DR_CAR")
todo <- if (length(args) > 2) args[-(1:2)] else CODES
logs <- list()
for (code in todo) {
  fd <- reader(code); r <- apply_rules(code, fd); lg <- r$log; lg$covariates <- ""; lg$covariate_source <- ""; lg$evaluated <- FALSE
  if (lg$N >= MIN_N && length(r$used)) {
    cv <- covariates(fd, r$L); lg$covariate_source <- cv$info; lg$covariates <- paste(names(cv$cov), collapse = " ")
    if (length(cv$cov)) {
      res <- data.frame(x = r$L$x, y = r$L$y, SOC = r$Y[, "SOC"], clay = r$Y[, "clay"], pH = r$Y[, "pH"], as.data.frame(cv$cov, check.names = FALSE),
                        check.names = FALSE)
      write.csv(res, file.path(out, paste0(code, "_points.csv")), row.names = FALSE); lg$evaluated <- TRUE
    }
  }
  logs[[code]] <- lg
  cat(code, ": ", lg$released, " released; rule 1 excluded ", lg$no_coordinates + lg$far + lg$shared, " (no coordinates ", lg$no_coordinates,
      ", far ", lg$far, ", shared ", lg$shared, "); corrupted records ", lg$corrupted, "; repeated values ", lg$repeated, " (", lg$repeat_by_sum,
      " by the 1000 g/kg sum); sampled ", lg$sampled, "; missing ", lg$missing, "; used: ", lg$properties, "; N = ", lg$N,
      if (lg$evaluated) paste0("; covariates (", lg$covariate_source, "): ", lg$covariates) else "; not evaluated", "\n", sep = "")
}
cols <- unique(unlist(lapply(logs, names)))
tab <- do.call(rbind, lapply(logs, function(l) as.data.frame(setNames(lapply(cols, function(k) if (is.null(l[[k]])) NA else l[[k]]), cols))))
write.csv(tab, file.path(out, if (length(args) > 2) "export_log_partial.csv" else "export_log.csv"), row.names = FALSE)
