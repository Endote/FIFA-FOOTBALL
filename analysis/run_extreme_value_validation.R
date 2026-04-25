#!/usr/bin/env Rscript

setwd('/Users/norbert.jaworski/Documents/small/WEC2026')

data_dir <- file.path(getwd(), "data")
output_dir <- file.path(getwd(), "output", "eda")

dir.create(output_dir, recursive = TRUE, showWarnings = FALSE)

runs_path <- file.path(data_dir, "player_appearance_run.csv")
quarters_path <- file.path(data_dir, "players_quarters_final.csv")

runs <- read.csv(runs_path, stringsAsFactors = FALSE)
quarters <- read.csv(quarters_path, stringsAsFactors = FALSE)

appearance_map <- unique(quarters[, c("player_appearance_id", "fixture_id", "date")])
run_events <- merge(
  runs,
  appearance_map,
  by = "player_appearance_id",
  all.x = TRUE,
  sort = FALSE
)

run_events$flag_distance_gt_1000 <- run_events$distance > 1000
run_events$flag_max_speed_gt_10_3 <- run_events$max_speed > 10.3
run_events$flag_any <- run_events$flag_distance_gt_1000 | run_events$flag_max_speed_gt_10_3

flag_reason <- character(nrow(run_events))
flag_reason[run_events$flag_distance_gt_1000] <- "distance_gt_1000"
speed_only <- !run_events$flag_distance_gt_1000 & run_events$flag_max_speed_gt_10_3
flag_reason[speed_only] <- "max_speed_gt_10_3"
both_flags <- run_events$flag_distance_gt_1000 & run_events$flag_max_speed_gt_10_3
flag_reason[both_flags] <- "distance_gt_1000; max_speed_gt_10_3"
run_events$flag_reason <- flag_reason

flagged_runs <- run_events[run_events$flag_any, c(
  "fixture_id", "date", "id", "player_appearance_id", "period", "minute",
  "stage", "possession", "run_type", "min_speed", "max_speed", "distance",
  "flag_distance_gt_1000", "flag_max_speed_gt_10_3", "flag_reason"
)]

flagged_runs <- flagged_runs[order(
  flagged_runs$fixture_id,
  flagged_runs$player_appearance_id,
  flagged_runs$period,
  flagged_runs$minute,
  flagged_runs$id
), ]

summary_by_fixture <- aggregate(
  cbind(
    flag_distance_gt_1000 = as.integer(run_events$flag_distance_gt_1000),
    flag_max_speed_gt_10_3 = as.integer(run_events$flag_max_speed_gt_10_3),
    flag_any = as.integer(run_events$flag_any)
  ),
  by = list(fixture_id = run_events$fixture_id, date = run_events$date),
  FUN = sum,
  na.rm = TRUE
)

summary_by_fixture <- summary_by_fixture[order(
  -summary_by_fixture$flag_any,
  summary_by_fixture$fixture_id
), ]

write.csv(
  flagged_runs,
  file.path(output_dir, "run_extreme_value_flags_R.csv"),
  row.names = FALSE
)

write.csv(
  summary_by_fixture,
  file.path(output_dir, "run_extreme_value_flags_R_summary.csv"),
  row.names = FALSE
)

cat("Flagged run events:", nrow(flagged_runs), "\n")
cat("Distance > 1000m:", sum(run_events$flag_distance_gt_1000, na.rm = TRUE), "\n")
cat("Max speed > 10.3 m/s:", sum(run_events$flag_max_speed_gt_10_3, na.rm = TRUE), "\n")
cat("\nTop fixtures by flagged run events:\n")
print(utils::head(summary_by_fixture, 10), row.names = FALSE)
