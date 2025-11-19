# Question 1

## create baseline Logistic PCR model in order to compare: 
## keeping HTML headers vs without headers

library(tidymodels)
library(modelr)
library(Matrix)
library(sparsesvd)
library(glmnet)

#################### NO HEADERS (BASELINE) ####################
source("scripts/preprocessing.R")

# load raw data
load("data/claims-raw.RData")

# preprocess (will take a minute or two)
claims_baseline <- claims_raw %>%
  parse_data() %>% 
  nlp_fn()

# partition
set.seed(110122)
partitions <- claims_baseline %>%
  initial_split(prop = 0.8)

# extract training and testing sets
train_baseline <- training(partitions)
test_baseline <- testing(partitions)

# separate predictors and response for baseline model
x_train <- train_baseline %>%
  select(-c(.id, bclass)) %>%
  as.matrix()

y_train <- train_baseline %>%
  pull(bclass) %>%
  as.numeric() - 1  # convert to 0/1

x_test <- test_baseline %>%
  select(-c(.id, bclass)) %>%
  as.matrix()

y_test <- test_baseline %>%
  pull(bclass) %>%
  as.numeric() - 1

# convert to sparse matrices for efficiency
x_train_sparse <- Matrix(x_train, sparse = TRUE)
x_test_sparse <- Matrix(x_test, sparse = TRUE)

# perform SVD for principal components
# choose number of components (e.g., 50, or adjust based on variance explained)
n_components <- 100

# compute SVD on training data
svd_result <- sparsesvd(x_train_sparse, rank = n_components)

# extract principal components
# PC scores for training data: U * D
pc_train <- svd_result$u %*% diag(svd_result$d)

# project test data onto principal components
# PC scores for test data: X_test * V
pc_test <- x_test_sparse %*% svd_result$v

# convert to data frames for glm
pc_train_df <- as.data.frame(pc_train)
pc_test_df <- as.data.frame(as.matrix(pc_test))

# fit logistic regression with ridge regularization to avoid overfitting
# without regularization warning messages of overfitting were displayed
# use cross-validation to select optimal lambda
set.seed(110122)
cv_baseline <- cv.glmnet(
  x = as.matrix(pc_train_df),
  y = y_train,
  family = "binomial",
  alpha = 0,  # ridge penalty (doesn't shrink PC coef's to 0)
  nfolds = 10
)

# fit final model with optimal lambda
baseline_pcr_model <- glmnet(
  x = as.matrix(pc_train_df),
  y = y_train,
  family = "binomial",
  alpha = 0,
  lambda = cv_baseline$lambda.min
)

# make predictions on test set
test_predictions <- predict(
  baseline_pcr_model,
  newx = as.matrix(pc_test_df),
  type = "response",
  s = cv_baseline$lambda.min
)

# calculate test accuracy
test_pred_class <- ifelse(test_predictions > 0.5, 1, 0)
baseline_accuracy <- mean(test_pred_class == y_test)

cat("Baseline (No Headers) Test Accuracy:", baseline_accuracy, "\n") # 0.797

#################### HEADERS (NEW MODEL) ####################

# preprocess (KEEPING HTML HEADERS)
claims_headers <- claims_raw %>%
  parse_data2() %>%
  nlp_fn()

# partition (using same seed for fair comparison)
set.seed(110122)
partitions_headers <- claims_headers %>%
  initial_split(prop = 0.8)

# extract training and testing sets
train_headers <- training(partitions_headers)
test_headers <- testing(partitions_headers)

# separate predictors and response for headers model
x_train_h <- train_headers %>%
  select(-c(.id, bclass)) %>%
  as.matrix()

y_train_h <- train_headers %>%
  pull(bclass) %>%
  as.numeric() - 1  # convert to 0/1

x_test_h <- test_headers %>%
  select(-c(.id, bclass)) %>%
  as.matrix()

y_test_h <- test_headers %>%
  pull(bclass) %>%
  as.numeric() - 1

# convert to sparse matrices for efficiency
x_train_h_sparse <- Matrix(x_train_h, sparse = TRUE)
x_test_h_sparse <- Matrix(x_test_h, sparse = TRUE)

# perform SVD for principal components (same number for fair comparison)
svd_result_h <- sparsesvd(x_train_h_sparse, rank = n_components)

# extract principal components
# PC scores for training data: U * D
pc_train_h <- svd_result_h$u %*% diag(svd_result_h$d)

# project test data onto principal components
# PC scores for test data: X_test * V
pc_test_h <- x_test_h_sparse %*% svd_result_h$v

# convert to data frames for glm
pc_train_h_df <- as.data.frame(pc_train_h)
pc_test_h_df <- as.data.frame(as.matrix(pc_test_h))

# fit logistic regression with ridge regularization to avoid overfitting
# use cross-validation to select optimal lambda
set.seed(110122)
cv_headers <- cv.glmnet(
  x = as.matrix(pc_train_h_df),
  y = y_train_h,
  family = "binomial",
  alpha = 0,  # ridge penalty
  nfolds = 10
)

# fit final model with optimal lambda
headers_pcr_model <- glmnet(
  x = as.matrix(pc_train_h_df),
  y = y_train_h,
  family = "binomial",
  alpha = 0,
  lambda = cv_headers$lambda.min
)

# make predictions on test set
test_predictions_h <- predict(
  headers_pcr_model,
  newx = as.matrix(pc_test_h_df),
  type = "response",
  s = cv_headers$lambda.min
)

# calculate test accuracy
test_pred_class_h <- ifelse(test_predictions_h > 0.5, 1, 0)
headers_accuracy <- mean(test_pred_class_h == y_test_h)

cat("Headers Model Test Accuracy:", headers_accuracy, "\n") # 0.771

# compare the two models
cat("Baseline (No Headers) Accuracy:", baseline_accuracy, "\n")  # 0.797
cat("Headers Model Accuracy:", headers_accuracy, "\n")           # 0.771

# RESULTS: binary class predictions are not improved when header information is
# added to logistic principle component regression