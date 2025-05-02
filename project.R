################################################################################
########################## Setup & data import #################################
################################################################################
library(tidyverse)
library(tidymodels)
#library(ggplot2)
library(magrittr)
library(caret)
library(gridExtra)
library(GGally)
library(pROC)
library(corrplot)
library(ggmosaic)
library(dplyr)
library(MASS)
library(randomForest)
library(rpart)
library(rpart.plot)
library(patchwork) # combining plots
library(car)
library(latexpdf)
library(rsample)
library(gbm)      # Gradient Boosting Machines
library(shiny)


getwd()
data <- read.csv("alzheimers_disease_data.csv")



################################################################################
############################ Hypothesis formulation ############################
################################################################################
# H0: After controlling for non-modifiable risk (Age, Gender, FamilyHistoryAlzheimers, HeadInjury) and cognitive status (MMSE, FunctionalAssessment), the composite Lifestyle Risk Index (LRI) is not associated with Alzheimer’s diagnosis.

# H1: A higher LRI is associated with higher odds of an Alzheimer’s diagnosis, independent of age, gender, family history, head injury, MMSE, and functional status.



################################################################################
############################## Initial Data Exploration ########################
################################################################################
# Drop pointless columns like the DoctorInCharge since it is empty and no use to us and PatientID.
data <- data %>% dplyr::select(-DoctorInCharge, -PatientID)

#Manual look at data
str(data)     # general info about columns (type...)
summary(data) # info about each column (min, max, median...)

sum(is.na(data))                  # how many are NAs in ds
colSums(is.na(data))              # how many are NAs in each column

# Check for duplicated rows
any( duplicated(data) )
# No duplicated values


# Check outcome balance
table(data$Diagnosis)             # AD diagnosis; 0 - No, 1 - Yes
prop.table(table(data$Diagnosis)) # AD diagnosis by percentage
# Dataset has more negatives diagnosis than positive (65%)


# Power check - events-per-variable (EPV) for logistic
events_pos <- sum(data$Diagnosis == 1)   # 760AD
p          <- 7                          # of covariates (measurable variable that has a statistical relationship with the dependent variable) in H1 model
EPV        <- events_pos / p
cat("Events-per-variable (EPV) =", round(EPV, 1), "\n")
# EPV is 108 which is > 10 which is generally considered adequate for stable logistic estimates

# Convert low-cardinality (few unique values) numerics to factors
for (col_name in names(data)) {
  # Check if column type is numeric and has less than 6 unique values
  if (is.numeric(data[[col_name]]) && length(unique(data[[col_name]])) <= 5) {
    data <- data %>%
      # unquote string to process as symbol (was headache without it)
      mutate(!!sym(col_name) := as.factor(!!sym(col_name)))
  }
}



num_cols <- data %>% dplyr::select(where(is.numeric), -Diagnosis) %>% names()

detect_outliers <- function(column) {
  q1 <- quantile(column, 0.25, na.rm = TRUE)
  q3 <- quantile(column, 0.75, na.rm = TRUE)
  iqr <- q3 - q1
  lower_bound <- q1 - 1.5 * iqr
  upper_bound <- q3 + 1.5 * iqr
  return(column < lower_bound | column > upper_bound)
}

# Apply to each numeric column and report
outlier_summary <- sapply(num_cols, function(colname) {
  sum(detect_outliers(data[[colname]]), na.rm = TRUE)
})

# Convert to data frame for easier viewing
outlier_df <- data.frame(Variable = names(outlier_summary),
                         Outlier_Count = as.integer(outlier_summary))

print(outlier_df)

# There are no outliers in numeric data

# VIF (Variance Inflation Factor for the logistic model
vif_vals <- car::vif(glm(Diagnosis ~ Age + Gender + FamilyHistoryAlzheimers +
                           HeadInjury + MMSE + FunctionalAssessment,
                         data = data, family = binomial))
print(vif_vals)

numeric_vars <- data %>% 
  select_if(is.numeric)
#' Normality
normality_p_values <- apply(numeric_vars, 2, function(x) {
  shapiro.test(x)$p.value
})
print(sort(normality_p_values, decreasing = T))
# P-values < 0.05 suggest a deviation from normality. Consider transformations if needed.
# we can observe that none or numeric attributes in normally distributed.

# Correlation for numeric variables
# TODO: enhance this for all keeping just the triangle- note vars are factorized 
# therefore not working.
cor_matrix <- cor(numeric_vars, use = "complete.obs")
corrplot(cor_matrix, 
         method = "color", 
         type = "upper", 
         order = "hclust",
         tl.col = "black", 
         tl.cex = 0.7,
         addCoef.col = "black",
         number.cex = 0.5,
         title = "Correlation Matrix of Numerical Variables",
         mar = c(0, 0, 2, 0))




# Check for outliers before deciding on transformations
data %>%
  select_if(is.numeric) %>%
  pivot_longer(everything(), names_to = "variable", values_to = "value") %>%
  ggplot(aes(x = value)) +
  geom_histogram(fill = "lightblue", color = "grey20") +
  facet_wrap(~ variable, scales = "free_x") +
  labs(title = "Histograms of Numerical Attributes", x = "Value", y = "Frequency") +
  theme_minimal()

# Check for categorical distributions
data %>%
  select_if(is.factor) %>%
  pivot_longer(everything(), names_to = "variable", values_to = "value") %>%
  ggplot(aes(x = value, fill = value)) +
  geom_bar() +
  facet_wrap(~ variable, scales = "free_x") +
  labs(title = "Bar Plots of Factor Attributes", x = "Category", y = "Count") +
  theme_minimal() +
  theme(axis.text.x = element_text(angle = 45, hjust = 1))

# EDA findings
# -No missing values in any column -> no imputation ( substituting missing data with a different value) needed
# -Slight diagnosis imbalance (65 % negative, 35 % positive) -> we will most likely adjust probability threshold to avoid bias
# -Numeric attributes - mostly uniform -> we keep all rows and skip outlier deletion
# -Categorical factors - most are highly imbalanced with the 0 level dominant
# -- Conclusion - data quality and distributions are suitable for the planned logistic and tree/forest models. No transformations or outlier removals are mandated at this stage.



################################################################################
################## 4. Requirements Analysis & Model Choice #####################
################################################################################
# R-1 picked candidate model families 
# We need algorithms that can handle:
# -binary outcome (Diagnosis 0 / 1),
# -a mix of numeric and factor predictors,
# -no missing data,
# -~2 000 observations  vs  ≤ 30 predictors.
#
# Methods:
# 1) Logistic GLM 
#     –Parametric, interpretable odds ratios
# 2) Linear Discriminant Analysis (LDA)
#     -Assumes multivariate-normal (normal distribution to two or more variables) predictors in each class
# 3) Decision Tree (rpart)
#     -Handles mixed types, captures non-linear splits, transparent rules
# 4) Random Forest
#     -Ensemble of trees: should yield higher AUC, robust to noise, accepts
#          both numeric and factor predictors out of the box
#
# These four (GLM, LDA, Tree, RF) will be evaluated against the dataset

candidate_models <- c("Logistic GLM",
                      "LDA",
                      "Decision Tree",
                      "Random Forest")

# R-2  Model–requirement matrix

# Dataset stats computed in EDA
outcome_binary <- (length(unique(data$Diagnosis)) == 2)
mixed_types <- any(sapply(data, is.factor)) & any(sapply(data, is.numeric))
missing_pct <- 100 * sum(is.na(data)) / prod(dim(data))
epv <- 108
vif_max <- max(vif_vals)

yes <- "Yes"
no  <- "No"

# Build the requirement table 
req_tab <- data.frame(
  Requirement    = c("Binary outcome", # model must handle 0 / 1
                     "Mixed num+factor OK", # both numeric & factor predictors present
                     "0 % missing OK", # we have 0 % NAs
                     "Distributional assumption met", # linear log-odds, MV-normal
                     sprintf("VIF ≤ 5  (max %.2f)", vif_max), # collinearity tolerance
                     sprintf("EPV ≥ 10  (EPV %d)", epv)), # sample-size rule
  
  Logistic_GLM   = c(yes, # GLM natively supports binomial family
                     yes, # factors become dummies automatically
                     if (missing_pct==0) yes else no, # all rows complete -> OK
                     "Linear log-odds", # GLM requires linearity in logit -> will be tested
                     if (vif_max<5) yes else no, # good value VIF cut-off
                     if (epv>=10)  yes else no), # EPV computed earlier (=108)
  
  LDA            = c(yes, # binary groups fine
                     no, # needs all-numeric
                     if (missing_pct==0) yes else no, # missing predictors
                     "MV-normal", # assumes multivariate normality
                     if (vif_max<10) yes else no, # LDA tolerates moderate VIF
                     if (nrow(data)>200) yes else no), # heuristic sample rule
  
  Decision_Tree  = c(yes, yes, yes, "None", yes, yes), # trees have no linearity or VIF limits
  
  Random_Forest  = c(yes, yes, yes, "None", yes, yes), # shares tree properties
  
  row.names = NULL, check.names = FALSE
)

print(req_tab)


# Model-selection rationale
# Data facts already known
cat("\nKey stats → EPV =", epv,
    "  | max VIF =", round(vif_max,2),
    "  | missing % =", missing_pct, "\n")

# Comparison
# Logistic GLM:
#  -Provides odds ratios that test H1 directly. 
#  -All assumptions met (binary outcome, EPV>10, VIF≈1, 0% missing).

# Decision-Tree: 
#  -Handles mixed predictors, no distributional assumptions, produces transparent IF-THEN rules. 

# Random-Forest:
#  -Likely higher raw accuracy than a single tree, but interpretability falls.
#  -We keep it only for feature-importance exploration, not as a deployable model.

# LDA: 
#  -Dropped - requires all-numeric and multivariate normality, both were violated here. 
# -Would add no value beyond GLM.

# Planned modeling path
# primary_model <- Logistic GLM: for inference / H1 test
# secondary_model <- Decision Tree: benchmark with explanations
# exploratory_model <- Random Forest: optional accuracy check



# Verify that Age, MMSE and FunctionalAssessment are approximately linearly related to the log-odds of Alzheimer’s (GLM assumption)
pred_vars <- c("Age", "MMSE", "FunctionalAssessment") # predictors to inspect

# Plot empirical logit for one variable 
emp_logit_plot <- function(df, xvar){
  tmp <- df %>% 
    
    # Slice predictor into 10 equal-width bins
    mutate(bin = cut(.data[[xvar]], breaks = 10)) %>% 
    group_by(bin) %>%
    
    summarise(
      # Compute bin midpoint
      mid = mean(.data[[xvar]]),
      # Diagnosis was converted to factor earlier, so turn back to 0/1
      pr  = (sum(as.numeric(as.character(Diagnosis))) + 0.5) / (n() + 1)
    ) %>% ungroup()
  
  # Scatter plot
  ggplot(tmp, aes(mid, log(pr/(1 - pr)))) +
    geom_point() +
    geom_smooth(method = "loess", se = FALSE, span = 1) +
    labs(title = paste("Empirical logistic -", xvar),
         # logistic(Pr(AD)) = models the log-odds of an event
         x = xvar, y = "logistic(Pr(AD))") +
    theme_minimal()
}

# Generate one plot per numeric predictor
for(v in pred_vars) print(emp_logit_plot(data, v))


# Results:
# Age                
#  Log-odds form a shallow U-shape: decrease up to ~70 y, flatten, then rise.

# MMSE
#  –Flat until ~12 points, then steep drop – strong non-linearity at low scores.

# FunctionalAssessment
#  –Smooth monotonic decline but still curved – mild non-linearity.



################################################################################
############################# 5. Data Spending #################################
################################################################################
# We pick one yechnique that: 
#  -Gives reliable performance estimates for our dataset  
#  -Keeps the 65 / 35 class ratio in every split  
#  -Is light-weight to run GLM, Tree and RF models repeatedly
#
# Chosen method:  
#  -STRATIFIED 10-fold cross-validation
#  –Ten folds - roughly 1 900 rows for training and roughly 215 for testing each round.  
#  –With 760 positive cases total, every fold still has roughly 75 AD patients
#   --metrics such as Sensitivity should be stable.  
#  –Computational cost is slight (10 fits per model) yet variance is much lower than a single 70 / 30 split ((estimate won't be overly optimistic or pessimistic)).



set.seed(65778275) # reproducible folds
cv_splits <- vfold_cv(
  data, 
  v      = 10, # 10-fold
  strata = Diagnosis  # keeps class balance in each fold
)

print(cv_splits)
# rsample object with 10 splits and class-stratification information














ggplot(data, aes(x = Gender, fill = Diagnosis)) +
  geom_bar(position = "dodge") +
  labs(title = "Gender vs. Diagnosis", x = "Gender", y = "Count") +
  theme_minimal() +
  scale_x_discrete(labels = c("Male", "Female"))
#' no significant change there

ggplot(data, aes(x = Diagnosis, y = Age, fill = Diagnosis)) +
  geom_boxplot() +
  labs(title = "Age vs. Diagnosis", x = "Diagnosis", y = "Age") +
  theme_minimal() +
  scale_x_discrete(labels = c("No Alzheimer's", "Alzheimer's"))
#' no significant change there


ggplot(data, aes(x = Smoking, fill = Diagnosis)) +
  geom_bar(position = "dodge") +
  labs(title = "Smoking vs. Diagnosis", x = "Smoking", y = "Count") +
  theme_minimal() +
  scale_x_discrete(labels = c("No", "Yes")) # Assuming 0=No, 1=Yes

ggplot(data, aes(x = Diagnosis, y = PhysicalActivity, fill = Diagnosis)) +
  geom_boxplot() +
  labs(title = "Physical Activity vs. Diagnosis", x = "Diagnosis", y = "Physical Activity (hours/week)") +
  theme_minimal() +
  scale_x_discrete(labels = c("No Alzheimer's", "Alzheimer's"))

ggplot(data, aes(x = Diagnosis, y = DietQuality, fill = Diagnosis)) +
  geom_boxplot() +
  labs(title = "Diet Quality vs. Diagnosis", x = "Diagnosis", y = "Diet Quality (score 0-10)") +
  theme_minimal() +
  scale_x_discrete(labels = c("No Alzheimer's", "Alzheimer's"))

ggplot(data, aes(x = Diagnosis, y = SleepQuality, fill = Diagnosis)) +
  geom_boxplot() +
  labs(title = "Sleep Quality vs. Diagnosis", x = "Diagnosis", y = "Sleep Quality (score 4-10)") +
  theme_minimal() +
  scale_x_discrete(labels = c("No Alzheimer's", "Alzheimer's"))
#' there is a slight difference


ggplot(data, aes(x = Diagnosis, y = BMI, fill = Diagnosis)) +
  geom_boxplot() +
  labs(title = "BMI vs. Diagnosis", x = "Diagnosis", y = "BMI") +
  theme_minimal() +
  scale_x_discrete(labels = c("No Alzheimer's", "Alzheimer's"))

ggplot(data, aes(x = Diagnosis, y = MMSE, fill = Diagnosis)) +
  geom_boxplot() +
  labs(title = "MMSE vs. Diagnosis", x = "Diagnosis", y = "MMSE Score") +
  theme_minimal() +
  scale_x_discrete(labels = c("No Alzheimer's", "Alzheimer's"))
#' this is MMSE (Cognitive Assessment), it appears to be a predictor on which
#' diagnosis depends the most so far.


















################################################################################
################# Data preparation & Lifestyle-Risk-Index (LRI) ################
################################################################################
# Goal is to keep only variables needed for the hypothesis + create a trimmed data frame for modelling

# Keep only variables required by the hypothesis
core_vars <- c(
  "Diagnosis",
  # non-modifiable risk factors
  "Age", "Gender", "FamilyHistoryAlzheimers", "HeadInjury",
  # baseline cognitive status
  "MMSE", "FunctionalAssessment",
  # lifestyle components
  "PhysicalActivity", "DietQuality", "SleepQuality",
  "Smoking", "AlcoholConsumption", "BMI"
)


#' Perform t-tests to compare means between diagnosis groups - 
#' draw conslusion from statistics now instead of visual boxplots 



categorical_vars <- c("Gender", "Ethnicity", "EducationLevel", "Smoking", 
                      "FamilyHistoryAlzheimers", "CardiovascularDisease", 
                      "Diabetes", "Depression", "Diagnosis")

health_metrics <- c("BMI", "SystolicBP", "DiastolicBP", "CholesterolTotal", 
                    "CholesterolLDL", "CholesterolHDL", "MMSE", "FunctionalAssessment", 
                    "SleepQuality")


t_test_results <- data.frame(
  Metric = character(),
  t_statistic = numeric(),
  p_value = numeric(),
  significant = character(),
  stringsAsFactors = FALSE
)
for (metric in health_metrics) {
  t_result <- t.test(formula(paste(metric, "~ Diagnosis")), data = data)
  
  t_test_results <- rbind(t_test_results, data.frame(
    Metric = metric,
    t_statistic = round(t_result$statistic, 3),
    p_value = round(t_result$p.value, 5),
    significant = ifelse(t_result$p.value < 0.05, "Yes", "No"),
    stringsAsFactors = FALSE
  ))
}

print(t_test_results)

key_corr_vars <- c("Age", "BMI", "MMSE", "FunctionalAssessment", 
                   "SystolicBP", "DiastolicBP", "CholesterolTotal", 
                   "PhysicalActivity", "AlcoholConsumption")

# Chi-square tests for categorical variables vs Diagnosis
chi_test_results <- data.frame(
  Variables = character(),
  chi_square = numeric(),
  p_value = numeric(),
  significant = character(),
  stringsAsFactors = FALSE
)

cat_vars <- c("Gender", "Ethnicity", "EducationLevel", "Smoking", 
              "FamilyHistoryAlzheimers", "CardiovascularDisease", 
              "Depression", "HeadInjury", "MemoryComplaints",
              "BehavioralProblems", "Confusion", "Disorientation",
              "PersonalityChanges", "DifficultyCompletingTasks", "Forgetfulness")

for (var in cat_vars) {
  chi_result <- chisq.test(data[[var]], data$Diagnosis)
  
  chi_test_results <- rbind(chi_test_results, data.frame(
    Variables = paste(var, "vs Diagnosis"),
    chi_square = round(chi_result$statistic, 3),
    p_value = round(chi_result$p.value, 5),
    significant = ifelse(chi_result$p.value < 0.05, "Yes", "No"),
    stringsAsFactors = FALSE
  ))
}

print(chi_test_results)




data_hyp <- data %>% dplyr::select(all_of(core_vars))

# Change each lifestyle component into a binary risk flag
# (1 = risky level, 0 = healthy/reference level)
data_hyp <- data_hyp %>% mutate(
  pa_risk      = ifelse(PhysicalActivity   <  3, 1, 0),  # < 3 h moderate PA / wk
  diet_risk    = ifelse(DietQuality        <= 4, 1, 0),  # low diet-quality score
  sleep_risk   = ifelse(SleepQuality       <= 5, 1, 0),  # poor sleep quality
  smoke_risk   = ifelse(Smoking            == 1, 1, 0),  # current smoker
  alcohol_risk = ifelse(AlcoholConsumption  > 14,1, 0),  # > 14 units / wk
  bmi_risk     = ifelse(BMI                >= 30,1, 0)   # obese
)

# Sum the six flags to obtain the Lifestyle-Risk-Index (LRI)
data_hyp <- data_hyp %>% mutate(
  LRI = pa_risk + diet_risk + sleep_risk + smoke_risk + alcohol_risk + bmi_risk
)

# Final fix of factor levels & drop rows with any missing values
data_hyp <- data_hyp %>% 
  mutate(
    Diagnosis = factor(Diagnosis, levels = c(0, 1)),  # ensure 0 = No, 1 = Yes
    Gender    = factor(Gender)                        # guarantee factor type
  ) %>% 
  drop_na()

################################################################################
################################ Hypothesis test ###############################
################################################################################
# H1: Higher LRI = higher odds of Alzheimer.

model_primary <- glm(
  Diagnosis ~ LRI +                                        # predictor
    Age + Gender + FamilyHistoryAlzheimers + HeadInjury +  # non-modifiable risk
    MMSE + FunctionalAssessment,                           # cognitive status
  data   = data_hyp,
  family = binomial
)

# Summarise regression
summary(model_primary)


model_secondary <- glm(
  Diagnosis ~ MMSE + FunctionalAssessment,
  data   = data_hyp,
  family = binomial
)

summary(model_secondary)

AIC(model_primary, model_secondary)
# AIC index shows that priamry/first model even having more variables is better.

# Multicolinearity
vif(model_secondary)
vif(model_primary)
# VIF values are < 5 therefore not multicolinearity detected.

# Convert log-odds to odds-ratios
exp(cbind(OR = coef(model_primary),   # exponentiate → odds ratios
          confint(model_primary)))    # 95 % CIs for interpretation

library(leaps)
m1_fit <- regsubsets(Diagnosis~ LRI +
                       Age + Gender + FamilyHistoryAlzheimers + HeadInjury +
                       MMSE + FunctionalAssessment, data=data_hyp)
summary(m1_fit)

# This resulting tables shows best combination of predictors for N number of
# predictors e.g. if we want just 1 - we pick FunctionalAssessment, for 2 predictors
# we pick FunctionalAssessment and MMSE, and for 3 we pick
# FunctionalAssessment, MMSE and FamilyHistoryAlzheimers1



#' Stepwise Selection to identify key predictors
#step_model <- stepAIC(full_model, direction = "both", trace = FALSE)
#final_formula <- formula(step_model)
#print("Variables selected by stepwise selection:")
#print(final_formula)











# Step 2: Split dataset into training and testing sets
set.seed(123)
train_index <- createDataPartition(data_hyp$Diagnosis, p = 0.7, list = FALSE)
train_data <- data_hyp[train_index, ]
test_data  <- data_hyp[-train_index, ]

# Step 3: Fit logistic regression model
model_primary <- glm(
  Diagnosis ~ LRI + Age + Gender + FamilyHistoryAlzheimers + HeadInjury + MMSE + FunctionalAssessment,
  data = train_data,
  family = binomial
)

summary(model_primary)  # Optional: check summary

# Step 4: Predict probabilities on test set
test_probs <- predict(model_primary, newdata = test_data, type = "response")

# Step 5: ROC analysis
roc_obj <- roc(test_data$Diagnosis, test_probs)
auc_val <- auc(roc_obj)
cat("AUC:", round(auc_val, 3), "\n")

# Step 6: Get optimal threshold (Youden's J)
coords_obj <- coords(roc_obj, "best", best.method = "youden", ret = c("threshold", "sensitivity", "specificity"))
optimal_threshold <- coords_obj["threshold"]
optimal_threshold

# Step 7: Confusion matrix - threshold 0.5
preds_default <- factor(ifelse(test_probs > 0.5, 1, 0), levels = c(0, 1))
conf_matrix_default <- confusionMatrix(preds_default, test_data$Diagnosis, positive = "1")
cat("\nConfusion Matrix (Threshold 0.5):\n")
print(conf_matrix_default)

# Step 8: Confusion matrix - optimal threshold
#preds_optimal <- factor(ifelse(test_probs > optimal_threshold, 1, 0), levels = c(0, 1))
#conf_matrix_optimal <- confusionMatrix(preds_optimal, test_data$Diagnosis, positive = "1")
#cat("\nConfusion Matrix (Optimal Threshold):\n")
#print(conf_matrix_optimal)

# Step 9: Plot ROC curve with optimal threshold marked
plot(roc_obj, main = "ROC Curve: Logistic Regression", col = "blue", lwd = 2)
points(1 - coords_obj["specificity"], coords_obj["sensitivity"], pch = 19, col = "red")
text(1 - coords_obj["specificity"], coords_obj["sensitivity"],
     labels = paste("Threshold =", round(optimal_threshold, 2)), pos = 4, col = "red")










################################################################################
############################ Hypothesis-test results ###########################
################################################################################
# Coefficient table shows:
# -LRI  ->  Est. =  –0.015   (p = 0.74) ... meaning NOT significant
# -Age  ->  Non-sig.
# -MMSE & FunctionalAssessment ... highly significant (***)
# --After we account for MMSE and FunctionalAssessment, the six-item lifestyle score does NOT add explanatory power -> Cognitive status dominates
# -Null deviance: 2792.3 on 2148: This high number tells us that it fits poorly. 
# Residual deviance: 2358.5 on 2141: Shows improvement with collective effort of predictors.



























################################################################################
########################### Alternative models (DT & RF) #######################
################################################################################
# X-fold cross-validation function that works for these 3 models:
# -glm – logistic regression
# -rpart – single decision tree
# -rf – random forest
# --It returns mean Accuracy, Sensitivity, Specificity, and AUC

# Args:
# -model_type (glm, rpart, rf)
# -formula_or_data (for GLM we need a formula, for tree models we ignore it)
# -data (must contain predictors and diagnosis)

evaluate_model <- function(model_type = c("glm", "rpart", "rf"),
                           formula_or_data, data){
  # For inner peace
  model_type <- match.arg(model_type) # ensure the supplied string is valid
  set.seed(65778275)                  # reproducibility
  
  # Make folds. Keeps class distribution equal-ish in each fold
  # k value can be experimented with
  folds <- createFolds(data$Diagnosis, k = 10)
  
  # Var for fold statistics
  cv_res <- data.frame(Accuracy = numeric(),
                       Sensitivity = numeric(),
                       Specificity = numeric(),
                       AUC = numeric())
  
  # Cross validation loop for index in folds
  for(idx in folds){
    # train set = all rows not in current fold -> model learns here
    # test set = current fold -> model is evaluated on unseen data
    train <- data[-idx, ]
    test <- data[idx, ]
    
    # Fit model
    # switch() chooses the fitting code block based on model_type
    fit <- switch(model_type,
                  glm = glm(formula_or_data, 
                            data = train, 
                            family = binomial), # (0 = No AD, 1 = AD)
                  
                  rpart = rpart(Diagnosis ~ ., 
                                data = train, 
                                method = "class", # use classification
                                xval = 0,         # disable build in 10x (we use 10 fold)
                                cp = 0.01),       # shallow base -> tune later!
                  
                  rf = randomForest(Diagnosis ~ ., 
                                    data = train,
                                    ntree = 500,        # pretty value, can be experimented with
                                    importance = TRUE)) # for feature importance plot
    
    # Predict probabilities on test data
    if(model_type == "glm"){
      prob <- predict(fit, test, type = "response") # numeric
    } else if (model_type == "rpart") {
      pmat <- predict(fit, test, type = "prob")
      
      # if the training fold had no positives, pmat has only column "0" -> resulting in error
      prob <- if ("1" %in% colnames(pmat)) pmat[, "1"] else rep(0, nrow(test))
    } else {# random-forest
      
      pmat <- predict(fit, test, type = "prob")
      prob <- pmat[, "1"]
    }
    # Convert probs to class labels using the canonical 0.5 cut-off
    # If value is >0.5 its 1 otherwise its 0
    pred <- factor(ifelse(prob > 0.5, 1, 0), levels = c(0,1))
    
    # Compute fold metrics
    # test$Diagnosis -> observed class values for this test split
    cm  <- confusionMatrix(pred, test$Diagnosis, positive = "1")
    
    # as.numeric produces either 1 or 2 so we substract 1 to get 0/1 -> roc need 0/1 values
    roc <- roc(as.numeric(test$Diagnosis) - 1, prob)
    
    # Store this fold’s metrics in cv_res
    cv_res <- rbind(cv_res, data.frame(
      Accuracy    = cm$overall["Accuracy"],    # (TP + TN) / total
      Sensitivity = cm$byClass["Sensitivity"], # TP / (TP + FN)
      Specificity = cm$byClass["Specificity"], # TN / (TN + FP)
      AUC         = auc(roc)
    ))
  }
  # Return mean metrics across the 10 folds
  colMeans(cv_res)
}
# Decision-tree (rpart) 
dt_metrics <- evaluate_model("rpart", NULL, data_hyp)
print(dt_metrics)

# Visualise the tree
dt_model <- rpart(Diagnosis ~ ., data = data_hyp, method = "class",
                  xval = 10, cp = 0.01)
rpart.plot(dt_model, type = 3, extra = 104,
           main = "Decision Tree for Alzheimer Diagnosis")

# Random-Forest
rf_metrics <- evaluate_model("rf", NULL, data_hyp)
print(rf_metrics)

# Variable-importance plot
N_top <- 15 # N with most impact
set.seed(65778275)
rf_full <- randomForest(Diagnosis ~ ., data = data_hyp,
                        ntree = 500, importance = TRUE)
varImp <- importance(rf_full)[, "MeanDecreaseGini", drop = FALSE] %>%
  as.data.frame() %>%
  rownames_to_column("Variable") %>%
  arrange(desc(MeanDecreaseGini)) %>%
  slice_head(n = N_top) 

ggplot(varImp, aes(x = reorder(Variable, MeanDecreaseGini),
                   y = MeanDecreaseGini)) +
  geom_bar(stat = "identity", fill = "lightblue") +
  coord_flip() + theme_minimal() +
  labs(title = paste("Random-Forest: Top", N_top, "variables"),
       x = NULL, y = "Mean Decrease Gini")

# Compare with primary logistic model
log_metrics <- evaluate_model(
  "glm",
  Diagnosis ~ LRI + Age + Gender + FamilyHistoryAlzheimers +
    HeadInjury + MMSE + FunctionalAssessment,
  data_hyp)

perf_tbl <- rbind(Logistic = log_metrics,
                  DecisionTree = dt_metrics,
                  RandomForest = rf_metrics)
print(round(perf_tbl, 3))















# Encode target variable (ensure this is binary 0/1, looks good already)
data$Diagnosis_numeric <- ifelse(data$Diagnosis == 1, 1, 0)

# Split data
set.seed(123)
split_idx <- createDataPartition(data$Diagnosis_numeric, p = 0.8, list = FALSE)
train_data <- data[split_idx, ]
test_data  <- data[-split_idx, ]

# Remove the target and ID-like columns from training predictors
train_x <- train_data %>% dplyr::select(-Diagnosis, -Diagnosis_numeric)
test_x  <- test_data %>% dplyr::select(-Diagnosis, -Diagnosis_numeric)

# One-hot encode if needed (you said categorical variables are int-coded, so no factors)
# Here you might not need one-hot encoding unless your variables are factors
# If all are numeric/int, you can skip this and just use them directly

# Combine back for GBM
train_full <- cbind(train_x, Diagnosis_numeric = train_data$Diagnosis_numeric)

# Check for constant columns or near-zero variance
nzv <- nearZeroVar(train_full, saveMetrics = TRUE)
print(nzv[nzv$nzv == TRUE, ])

# Remove them if necessary
train_full <- train_full[, !nzv$nzv]

# Train GBM with correct formula
gbm_model <- gbm(
  Diagnosis_numeric ~ .,
  data = train_full,
  distribution = "bernoulli",
  n.trees = 100,
  shrinkage = 0.1,
  interaction.depth = 3,
  n.minobsinnode = 10,
  verbose = FALSE
)

# Check summary (variable importance)
summary(gbm_model)


# Remove NZV columns from test set using the same columns removed from train
test_x <- test_x[, colnames(test_x) %in% colnames(train_full)]

# Predict probabilities
pred_probs <- predict(gbm_model, newdata = test_x, n.trees = 100, type = "response")

# Convert probabilities to class labels (0 or 1) using 0.5 threshold
pred_labels <- ifelse(pred_probs > 0.5, 1, 0)

# Actual labels
true_labels <- test_data$Diagnosis_numeric

# Confusion matrix
conf_mat <- confusionMatrix(as.factor(pred_labels), as.factor(true_labels), positive = "1")
print(conf_mat)

# AUC
roc_obj <- roc(true_labels, pred_probs)
auc_value <- auc(roc_obj)
cat("AUC:", auc_value, "\n")

# Accuracy reached 95%

