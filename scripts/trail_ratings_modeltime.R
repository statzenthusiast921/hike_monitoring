#Let's try the modeltime process for alltrails ratings
#Date: August 9th, 2026
#----------------------------------------

#Load libraries ----
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
})

#Load data ----
message('Loading data...')
df <- read.csv("~/Desktop/Data Projects/AllTrails/data/synthetic_hiking_reviews.csv") |>
  mutate(
    date = as.Date(date),
    year_month = lubridate::floor_date(date, 'month'),
    year = lubridate::year(date),
    month = lubridate::month(date)
  )

max_actual <- max(df$year_month)

#Make sure each time series is complete - impute 0 for missing ----
message('Prepping the data...')

df <- df |>
  group_by(trail_name) |>
  pad_by_time(
    .date_var = year_month,
    .by = 'auto',
    .pad_value = 0,
    .start_date = min(df$year_month),
    .end_date = max(df$year_month)
  )

# ----- Create case weights - more recent day gets more weight
df <- df |>
  group_by(trail_name) |>
  mutate(
    dec_year = lubridate::decimal_date(year_month),
    case_wts = exp(dec_year - max(dec_year))
  )

df_monthly <- df |>
  group_by(trail_name, state, latitude, longitude, difficulty, elevation_gain_ft, year_month) |>
  summarise(avg_monthly_rating = mean(rating))
        

# ----- Extend each time series into the future
df_ext <- df_monthly |>
  group_by(trail_name) |>
  future_frame(
    .date_var = year_month,
    .length_out = '2 years',
    .bind_data = TRUE
  )


#Split into full training data and future data that will be forecasted ----
message('Make 2 partitions of data (full, future)...')

# ----- Full dataset
df_full_data <- df_ext |>
  drop_na() |>
  tidyr::nest(data_full = c(-trail_name))

df_future_data <-df_ext |>
  filter(is.na(avg_monthly_rating)==TRUE) |>
  tidyr::nest(data_future = c(-trail_name))


# ------ Join data all together
message('Join full and future data together in nested df...')

df_nest <- inner_join(
  df_full_data,
  df_future_data,
  by = 'trail_name'
)


# ------ Create training and calibration (test) data
message('Make 2 partitions of data (train, test)...')

df_nest <- df_nest |>
  mutate(
    splits = map(
      data_full, .f = function(x) {
        time_series_split(x, assess = 12, cumulative = TRUE)
      }
    )
  )



df_nest <- df_nest |>
  mutate(
    data_train = map(.x = data_full, .f = ~slice_head(.x, n = -12)),
    data_calib = map(.x = data_full, .f = ~slice_tail(.x, n =  12))
  ) |>
  relocate(data_train, .after=data_full) |>
  relocate(data_calib, .after=data_train)

# ------ Recipes
message('Define recipes (ie: model params)...')

rec_list <- list()

num_of_trails <- dim(df_nest)[1]

for (i in 1:num_of_trails) {
  rec_list[[i]] <- recipe(avg_monthly_rating ~ ., data = df_nest$data_train[[i]]) |>
    # 1. Create smooth sine/cosine waves for 12-month seasonality
    step_fourier(year_month, period = 12, K = 2) |>
    
    # 2. Remove the raw date column so lm() doesn't fail
    step_rm(year_month) |>
    
    # 3. Automatically drop static trail columns (elevation, difficulty, state, etc.)
    step_zv(all_predictors()) |>
    
    # 4. Dummy encode any remaining categorical predictors if needed
    step_dummy(all_nominal_predictors())
}

# ------ Workflows
message('Assign recipes to workflow...')

# ----- Linear Model 
model_lm <- linear_reg() |>
  set_engine("lm")
message('Assign recipes to workflows...')

wfl_list <- list()

for (i in 1:num_of_trails) {
  wfl_list[[i]] <- workflow() |>
    add_model(model_lm) |>
    add_recipe(rec_list[[i]])
}


message('Assign workflows to groups in nested df...')

df_nest$.wfl <- wfl_list

# -----Fit models
message('Fit workflows using training data...')

df_nest <- df_nest |>
  mutate(.fit = map2(.x = .wfl,
                     .y = data_train,
                     .f = ~fit(.x, .y)))


# ----- Calibrate models
message('Calibrate models using test data...')

df_nest <- df_nest |>
  mutate(.calib = future_map2(.x = .fit,
                              .y = data_calib,
                              .f = ~modeltime_calibrate(modeltime_table(.x),new_data=.y),
                              .options = furrr_options(packages = c("timetk","purrr"))))


# ----- Refit models
message('Refit models using all data...')

df_nest <- df_nest |>
  mutate(.refit = future_map2(.x = .calib,
                              .y = data_full,
                              .f = ~modeltime_refit(.x,data=.y)))


# ----- Generate forecasts
message('Generate forecasts...')

df_nest <- df_nest |>
  mutate(.fc = future_pmap(.l = list(.refit,data_future,data_full),
                           .f = ~modeltime_forecast(
                             object = ..1,
                             new_data = ..2,
                             actual_data = ..3,
                             keep_data = FALSE
                           ),.options = furrr_options(packages = c("timetk","purrr"))
  )
  )



message('Make predictions dataset by pulling out forecasts...')
preds_list <-list()

# ------ Make a predictions dataset
for (i in 1:num_of_trails){
  preds_list[[i]] <- df_nest[[11]][[i]] |>
    filter(.key == "prediction") |>
    mutate(
      year_month = .index,
      avg_monthly_rating = .value,
      trail_name = df_nest$trail_name[i]
      
    ) |>
    select(year_month, avg_monthly_rating, trail_name) 
  
}

preds_df <- purrr::list_rbind(preds_list)


df_final <- bind_rows(
  df_ext |> mutate(key = 'ACTUAL') |> filter(year_month < max_actual),
  preds_df |> rename(avg_monthly_rating_pred = avg_monthly_rating) |> mutate(key = 'PRED')
) |>
  mutate(
    avg_monthly_rating = coalesce(avg_monthly_rating, avg_monthly_rating_pred),
    avg_monthly_rating = ifelse(avg_monthly_rating == 0, NA, avg_monthly_rating), 
  ) |>
  select(-c(avg_monthly_rating_pred)) |>
  arrange(trail_name, year_month)


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
    color = "Series"
  ) +
  theme_minimal() +
  theme(
    strip.text = element_text(face = "bold", size = 9),
    legend.position = "bottom"
  )
