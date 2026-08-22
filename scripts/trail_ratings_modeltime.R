#Let's try the modeltime process for alltrails ratings
#Date: August 9th, 2026
#----------------------------------------

# ----- Load libraries
message('Loading packages...')
suppressPackageStartupMessages({
  library(tidyverse)
  library(tidymodels)
  library(modeltime)
  library(timetk)
  library(lubridate)
  library(timeDate)
  library(furrr)
  library(tictoc)
  library(tidyr)
  library(workflows)
  library(sentimentr)
  library(broom)
  library(hardhat)
})

message('Loading data...')

neg_bear_pattern <- paste(
  c("bear grass", "beargrass", "bearings", "bearable", "bear spray", 
    "no bear", "see any bears", "encounter any bears", 
    "bears – thankfully, we didn't spot any", "didn't see any bears", "no bears"),
  collapse = "|"
)



df <- read.csv("~/Desktop/Data Projects/AllTrails/data/synthetic_hiking_reviews.csv") |>
  mutate(
    date = as.Date(date),
    year_month = lubridate::floor_date(date, 'month'),
    sentiment = (pmin(pmax(sentimentr::sentiment_by(sentimentr::get_sentences(review_text))$ave_sentiment, -1), 1) + 1) / 2,
    sunny_flag = as.integer(str_detect(review_text, regex("\\b(sun|sunny|sunshine)\\b", ignore_case = TRUE))),
    rainy_flag = as.integer(str_detect(review_text, regex("\\b(rainy|rain)\\b", ignore_case = TRUE))),
    snowy_flag = as.integer(str_detect(review_text, regex("\\b(snowy|snow)\\b", ignore_case = TRUE))),
    cloudy_flag = as.integer(str_detect(review_text, regex("\\b(cloudy|cloud|clouds)\\b", ignore_case = TRUE))),
    wildlife_flag = as.integer(str_detect(review_text, regex("\\b(deer|hawk|hawks|rabbit|rabbits|pika|pikas|bird|birds|mosquito|mosquitos|mosquitoes|chipmunk|chipmunks|elk|elks|mountain goat|mountain goats)\\b", ignore_case = TRUE))),
    bear_flag = as.integer(
      str_detect(review_text, regex("\\b(bear|bears|grizzly|grizzlies)\\b", ignore_case = TRUE)) & 
        !str_detect(review_text, regex(neg_bear_pattern, ignore_case = TRUE))
    ),    
    raw_suffer = (1 - sentiment) * difficulty,
    suffer_index = ((raw_suffer - min(raw_suffer, na.rm = TRUE)) / 
                      (max(raw_suffer, na.rm = TRUE) - min(raw_suffer, na.rm = TRUE))) * 4 + 1
  )

df_monthly <- df |>
  group_by(trail_name, year_month) |>
  summarise(
    avg_monthly_rating = mean(rating, na.rm = TRUE),
    agg_reviews = paste(review_text[!is.na(review_text) & review_text != ""], collapse = " "),
    med_sentiment = median(sentiment, na.rm = TRUE),
    tot_sunny = sum(sunny_flag, na.rm = TRUE),
    tot_cloudy = sum(cloudy_flag, na.rm = TRUE),
    tot_rainy = sum(rainy_flag, na.rm = TRUE),
    tot_snowy = sum(snowy_flag, na.rm = TRUE),
    tot_wildlife = sum(wildlife_flag, na.rm = TRUE),
    tot_bear = sum(bear_flag, na.rm = TRUE),
    med_suffer_index = median(suffer_index, na.rm = TRUE)
  )

max_actual <- max(df_monthly$year_month)

# ------ Add variable measures
message('Prepping the data...')

global_suffer_med <- median(df$suffer_index, na.rm = TRUE)

df_monthly <- df_monthly |>
  group_by(trail_name) |>
  pad_by_time(
    .date_var = year_month,
    .by = 'month',
    .pad_value = NA,
    .start_date = min(df_monthly$year_month),
    .end_date = max(df_monthly$year_month)
  ) |>
  mutate(
    tot_sunny = coalesce(tot_sunny, 0),
    tot_cloudy = coalesce(tot_cloudy, 0),
    tot_rainy = coalesce(tot_rainy, 0),
    tot_snowy = coalesce(tot_snowy, 0),
    tot_wildlife = coalesce(tot_wildlife, 0),
    tot_bear = coalesce(tot_bear, 0),
    med_suffer_index = coalesce(med_suffer_index, median(med_suffer_index, na.rm = TRUE), global_suffer_med)
  ) |>
  ungroup()

# ----- Create case weights
df_monthly <- df_monthly |>
  group_by(trail_name) |>
  mutate(
    dec_year = lubridate::decimal_date(year_month),
    case_wts = exp(dec_year - max(dec_year))
  ) |>
  ungroup()

# ----- Extend each time series into the future
df_ext <- df_monthly |>
  group_by(trail_name) |>
  future_frame(
    .date_var   = year_month,
    .length_out = '2 years',
    .bind_data  = TRUE
  ) |>
  mutate(
    across(
      c(tot_sunny, tot_cloudy, tot_rainy, tot_snowy, tot_wildlife, tot_bear, med_suffer_index),
      ~ if_else(is.na(.x), mean(.x, na.rm = TRUE), .x)
    )
  ) |>
  mutate(
    med_suffer_index = coalesce(med_suffer_index, global_suffer_med),
    across(c(tot_sunny, tot_cloudy, tot_rainy, tot_snowy, tot_wildlife, tot_bear), ~ coalesce(.x, 0)),
    case_wts = hardhat::importance_weights(coalesce(case_wts, 1))
  ) |>
  ungroup()

# ----- Split into full training data and future data
message('Make 2 partitions of data (full, future)...')

df_full_data <- df_ext |>
  filter(!is.na(avg_monthly_rating)) |>
  tidyr::nest(data_full = c(-trail_name))

df_future_data <- df_ext |>
  filter(is.na(avg_monthly_rating) == TRUE) |>
  tidyr::nest(data_future = c(-trail_name))

df_nest <- inner_join(
  df_full_data,
  df_future_data,
  by = 'trail_name'
)

# ------ Create training and calibration data
message('Make 2 partitions of data (train, test)...')

df_nest <- df_nest |>
  mutate(
    splits = map(
      data_full, .f = function(x) {
        time_series_split(x, date_var = year_month, assess = 12, cumulative = TRUE)
      }
    )
  )

df_nest <- df_nest |>
  mutate(
    data_train = map(.x = data_full, .f = ~slice_head(.x, n = -12)),
    data_calib = map(.x = data_full, .f = ~slice_tail(.x, n = 12))
  ) |>
  relocate(data_train, .after = data_full) |>
  relocate(data_calib, .after = data_train)

# ------ Recipes
message('Define recipes...')

rec_list <- list()
num_of_trails <- dim(df_nest)[1]

for (i in 1:num_of_trails) {
  rec_list[[i]] <- recipe(
    avg_monthly_rating ~ year_month + med_suffer_index + 
      tot_sunny + tot_cloudy + tot_rainy + tot_snowy + tot_wildlife + tot_bear + case_wts, 
    data = df_nest$data_train[[i]]
  ) |>
    step_fourier(year_month, period = 12, K = 2) |>
    step_rm(year_month) |>
    step_impute_median(all_numeric_predictors())
}

# ------ Workflows
model_lm <- linear_reg() |>
  set_engine("lm")

wfl_list <- list()
for (i in 1:num_of_trails) {
  wfl_list[[i]] <- workflow() |>
    add_model(model_lm) |>
    add_recipe(rec_list[[i]]) |>
    add_case_weights(case_wts)
}

df_nest$.wfl <- wfl_list

# -----Fit models
message('Fit workflows using training data...')
df_nest <- df_nest |>
  mutate(.fit = map2(.x = .wfl, .y = data_train, .f = ~fit(.x, .y)))

# ----- Calibrate models
message('Calibrate models using test data...')
df_nest <- df_nest |>
  mutate(.calib = future_map2(.x = .fit, .y = data_calib,
                              .f = ~modeltime_calibrate(modeltime_table(.x), new_data = .y),
                              .options = furrr_options(
                                packages = c("timetk", "purrr"), 
                                seed = TRUE)
  )
  )

# ----- Refit models
message('Refit models using all data...')
df_nest <- df_nest |>
  mutate(.refit = future_map2(.x = .calib, .y = data_full,
                              .f = ~modeltime_refit(.x, data = .y),
                              .options = furrr_options(
                                packages = c("timetk", "purrr"), seed = TRUE)
  )
  )

# ----- Generate forecasts
message('Generate forecasts...')
df_nest <- df_nest |>
  mutate(.fc = future_pmap(.l = list(.refit, data_future, data_full),
                           .f = ~modeltime_forecast(object = ..1, new_data = ..2, actual_data = ..3, keep_data = FALSE), 
                           .options = furrr_options(
                             packages = c("timetk", "purrr"), seed = TRUE)
  )
  )

# ----- Extract predictions
preds_list <- list()
for (i in 1:num_of_trails){
  preds_list[[i]] <- df_nest$.fc[[i]] |>
    filter(.key == "prediction") |>
    mutate(year_month = .index, avg_monthly_rating = .value, trail_name = df_nest$trail_name[i]) |>
    select(year_month, avg_monthly_rating, trail_name) 
}

preds_df <- purrr::list_rbind(preds_list)

df_final <- bind_rows(
  df_ext |> mutate(key = 'ACTUAL') |> filter(year_month <= max_actual & !is.na(avg_monthly_rating)),
  preds_df |> filter(year_month > max_actual) |> rename(avg_monthly_rating_pred = avg_monthly_rating) |> mutate(key = 'PRED')
) |>
  mutate(avg_monthly_rating = coalesce(avg_monthly_rating, avg_monthly_rating_pred)) |>
  select(-c(avg_monthly_rating_pred)) |>
  arrange(trail_name, year_month)

# ----- Plot Actuals vs. Forecasts
ggplot(df_final, aes(x = year_month, 
                     y = avg_monthly_rating, 
                     color = key, 
                     group = key)) +
  geom_line(linewidth = 0.8) +
  geom_point(size = 1.5) +
  facet_wrap(~ trail_name, ncol = 2, scales = "fixed") +
  labs(
    title = "Monthly Rating Forecasts by Trail",
    x = "Date",
    y = "Average Rating",
    color = "Key"
  ) +
  theme_minimal() +
  theme(
    strip.text = element_text(face = "bold", size = 9),
    legend.position = "bottom"
  )

# ----- Extract coefficients with p-values
trail_coefficients <- df_nest |>
  mutate(
    model_coefs = map(.refit, ~ {
      .x |>
        pluck(".model", 1) |>
        extract_fit_engine() |>
        tidy(conf.int = TRUE)
    })
  ) |>
  select(trail_name, model_coefs) |>
  unnest(model_coefs)


# ----- Write out final results
write.csv(
  trail_coefficients, 
  file = '/Users/jonzimmerman/Desktop/Data Projects/AllTrails/data/trail_model_estimates.csv', 
  row.names = FALSE
)

write.csv(
  df_final |> mutate(case_wts = as.numeric(case_wts)), 
  file = '/Users/jonzimmerman/Desktop/Data Projects/AllTrails/data/final_model_results.csv', 
  row.names = FALSE
)
