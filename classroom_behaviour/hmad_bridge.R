# Run upstream HMAD models and export stable, explicitly timed adapter tables.
# HMAD itself remains in a separate, pinned checkout with its original license.
args <- commandArgs(trailingOnly=TRUE)
if (length(args) != 3) stop("Usage: Rscript hmad_bridge.R HMAD_WORK_ROOT PROJECT_NAME RESULT_DIR")
HMAD_DIRECTORY <- normalizePath(args[1], mustWork=TRUE)
PROJECT_NAME <- args[2]
RESULT_DIRECTORY <- normalizePath(args[3], mustWork=TRUE)
TRACKING_SOFTWARE <- "OPENFACE"
verbose <- TRUE
setwd(HMAD_DIRECTORY)
source("sourceR/loadSourcesAndPackages.R")
source("sourceObjectR/loadObjectSourcesAndPackages.R")
# The upstream ELAN helper predates WebM. Extend only its MIME lookup.
upstream_mime_type <- lpl.R.dev.elan.getElanMimeType
lpl.R.dev.elan.getElanMimeType <- function(media_file) {
    if (grepl("\\.webm$", media_file, ignore.case=TRUE)) return("video/webm")
    upstream_mime_type(media_file)
}
project_directory <- file.path(HMAD_DIRECTORY, "projects", PROJECT_NAME)
media_path_file <- file.path(project_directory, "media_path.txt")
media_file <- if (file.exists(media_path_file)) readLines(media_path_file, n=1) else file.path(project_directory, "analysis.avi")
csv_file <- file.path(project_directory, "OPENFACE", "processed", paste0(PROJECT_NAME, "_ofo.csv"))
raw <- read.csv(csv_file, check.names=TRUE)
status <- data.frame(component=character(), status=character(), reason=character())

attempt <- function(name, work) {
    tryCatch({
        value <- work()
        status[nrow(status)+1, ] <<- c(name, "ok", "")
        value
    }, error=function(e) {
        cat("HMAD component", name, "failed:", conditionMessage(e), "\n")
        print(conditionCall(e))
        status[nrow(status)+1, ] <<- c(name, "error", conditionMessage(e))
        NULL
    })
}

export_intervals <- function(name, starts, ends, values) {
    d <- data.frame(start_s=as.numeric(starts), end_s=as.numeric(ends), value=as.character(values))
    write.table(d, file.path(RESULT_DIRECTORY, paste0(name, ".tsv")),
                sep="\t", quote=FALSE, row.names=FALSE, na="NA")
}

export_samples <- function(name, times, values) {
    if (length(times) == 0) stop(paste("No samples for", name))
    changes <- c(TRUE, values[-1] != values[-length(values)])
    starts <- which(changes)
    ends <- c(starts[-1], length(times)+1)
    boundaries <- c(times, tail(times, 1)+0.04)
    export_intervals(name, boundaries[starts], boundaries[ends], values[starts])
}

df <- attempt("head_model", function() {
    lpl.R.dev.hmad.createHeadModelAndResiduals(TRACKING_SOFTWARE, PROJECT_NAME)
})
audf <- attempt("action_units", function() {
    lpl.R.dev.hmad.createActionUnitTable(TRACKING_SOFTWARE, PROJECT_NAME)
})

attempt("smile", function() {
    if (is.null(df) || is.null(audf)) stop("Head model / action unit table unavailable")
    smad <- lpl.R.dev.smad.computeSMAD(TRACKING_SOFTWARE, PROJECT_NAME, df, audf)
    export_intervals("smile", smad$tmin, smad$tmax, smad$smile14)
    TRUE
})

attempt("brows", function() {
    if (is.null(df)) stop("Head model unavailable")
    lpl.R.dev.ebmad.computeEBMAD(TRACKING_SOFTWARE, PROJECT_NAME, df)
    folder <- file.path(project_directory, "OPENFACE", "EBMAD", "tables")
    for (pair in list(c("ebrmad", "brow_raising"), c("ebfmad", "brow_lowering"))) {
        d <- read.delim(file.path(folder, paste0(pair[1], ".txt")))
        export_intervals(pair[2], d$tmin, d$tmax, d$class)
    }
    TRUE
})

# Blink inference is independent of fitting a local gaze-reference region.
hmad_object <- list(
    df_landmark=data.frame(time=raw$timestamp, confidence=raw$confidence),
    df_au=raw,
    df_gaze=data.frame(gaze_angle_x=raw$gaze_angle_x*180/pi,
                       gaze_angle_y=raw$gaze_angle_y*180/pi))

blink_result <- attempt("bagmad_blinks", function() {
    object <- lpl.R.dev.hmad.bagmadObject.newBagmadObject()
    object <- lpl.R.dev.hmad.bagmadObject.createInput(object, hmad_object, TRUE)
    model <- lpl.R.dev.hmad.bagmadModelObject.newBagmadModelObject()
    model <- lpl.R.dev.hmad.bagmadModelObject.loadAndCreateBlinkModel(
        model, "hmm", file.path(HMAD_DIRECTORY, "models", "OPENFACE", "bagmad"))
    object <- lpl.R.dev.hmad.bagmadObject.setBagmadModelObject(object, model)
    object <- lpl.R.dev.hmad.bagmadObject.computeBlinkSolution(object, TRUE)
    blink <- object[[4]]
    values <- rep("no blink", nrow(raw))
    if (nrow(blink) > 0) {
        for (i in seq_len(nrow(blink))) {
            values[raw$timestamp >= blink$tmin[i] & raw$timestamp < blink$tmax[i]] <- "blink"
        }
    }
    export_samples("blinking", raw$timestamp, values)
    object
})

gaze_result <- attempt("bagmad_gaze", function() {
    if (is.null(blink_result)) stop("Blink solution unavailable")
    object <- lpl.R.dev.hmad.bagmadObject.loadAndCreateDefaultBagmadModel(
        hmad_object, HMAD_DIRECTORY, project_directory, TRUE)
    parameters <- object$model_object$local_model$df_parameters
    if (!all(is.finite(unlist(parameters))) || parameters$sd_x_ec <= 0 || parameters$sd_y_ec <= 0)
        stop("Degenerate local gaze-reference statistics")
    object[[4]] <- blink_result[[4]]
    object <- lpl.R.dev.hmad.bagmadObject.computeGazeSolution(object, TRUE, TRUE)
    lpl.R.dev.hmad.bagmadObject.writeDefaultBagmadOutput(object, project_directory, TRUE)
    predictions <- object[[5]]
    export_samples("gaze", predictions$time, predictions$prediction)
    object
})

attempt("elan", function() {
    if (is.null(gaze_result)) stop("Gaze solution unavailable for optional ELAN export")
    result_directory <- file.path(project_directory, "OPENFACE", "bagmad", "result")
    dir.create(result_directory, showWarnings=FALSE, recursive=TRUE)
    lpl.R.dev.hmad.bagmadObject.createBagmadElanOutput(gaze_result,
        media_file, file.path(result_directory, "bagmad.eaf"), TRUE)
    TRUE
})
status[nrow(status)+1, ] <- c("bagmad",
    if (!is.null(blink_result) && !is.null(gaze_result)) "ok" else "partial",
    "See separate blink and gaze component statuses")

write.table(status, file.path(RESULT_DIRECTORY, "status.tsv"), sep="\t",
            quote=TRUE, row.names=FALSE)
capture.output(sessionInfo(), file=file.path(RESULT_DIRECTORY, "R_session.txt"))
