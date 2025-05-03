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
library(patchwork) # combining plots
library(car)    # Levene's test for homoscedasticit
library(latexpdf)
library(rsample)  
library(purrr)
library(gbm)
library(randtests)
library(lmtest)
#library(shiny)

getwd()
data <- read.csv("alzheimers_disease_data.csv")



################################################################################
############################## Initial Data Exploration ########################
################################################################################
# Drop pointless columns like the DoctorInCharge since it is empty and no use to us and PatientID.
data <- data %>% dplyr::select(-DoctorInCharge, -PatientID)
data_numeric <- data
#Manual look at data
str(data)     # general info about columns (type...)
summary(data) # info about each column (min, max, median...)

sum(is.na(data)) # how many are NAs in ds
colSums(is.na(data)) # how many are NAs in each column

# Check for duplicated rows
any( duplicated(data) )

# Check outcome balance
table(data$Diagnosis)             # AD diagnosis; 0 - No, 1 - Yes
prop.table(table(data$Diagnosis)) # AD diagnosis by percentage
# Dataset has more negatives diagnosis than positive (65%)

# Power check - events-per-variable (EPV) for logistic
events_pos <- sum(data$Diagnosis == 1) # 760AD
p <- 7 # of covariates (measurable variable that has a statistical relationship with the dependent variable) in H1 model
EPV <- events_pos / p
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

numeric_attrs_names <- data %>% dplyr::select(where(is.numeric), -Diagnosis) %>% names()
factor_attrs_names <- data %>%  select_if(is.factor) %>% names()

numeric_attrs <- data %>% select_if(is.numeric)
factor_attrs <- data %>% select_if(is.factor)

detect_outliers <- function(column) {
  q1 <- quantile(as.numeric(column), 0.25, na.rm = TRUE)
  q3 <- quantile(as.numeric(column), 0.75, na.rm = TRUE)
  iqr <- q3 - q1
  lower_bound <- q1 - 1.5 * iqr
  upper_bound <- q3 + 1.5 * iqr
  return(column < lower_bound | column > upper_bound)
}
outlier_summary <- sapply(colnames(data_numeric), function(colname) {
  sum(detect_outliers(data[[colname]]), na.rm = TRUE)
})
outlier_df <- data.frame(Variable = names(outlier_summary),
                         Outlier_Count = as.integer(outlier_summary))
print(outlier_df)

# TODO: toto az do vyhodnocovania modelu, vymazat, ale pozor lebo to je nizsie pouzita premenna
# VIF (Variance Inflation Factor for the logistic model
vif_vals <- car::vif(glm(Diagnosis ~ Age + Gender + FamilyHistoryAlzheimers +
                           HeadInjury + MMSE + FunctionalAssessment,
                         data = data, family = binomial))
print(vif_vals)
# There are no outliers in numeric data

#' Normality
normality_p_values <- apply(numeric_attrs, 2, function(x) {
  shapiro.test(x)$p.value
})
print(sort(normality_p_values, decreasing = T))
# P-values < 0.05 suggest a deviation from normality. Consider transformations if needed.
# we can observe that none or numeric attributes in normally distributed.


data %>%
  select_if(is.numeric) %>%
  pivot_longer(everything(), names_to = "variable", values_to = "value") %>%
  ggplot(aes(x = value)) +
  geom_histogram(fill = "lightblue", color = "grey20") +
  facet_wrap(~ variable, scales = "free_x") +
  labs(title = "Histograms of Numerical Attributes", x = "Value", y = "Frequency") +
  theme_minimal()

numeric_data <- bind_cols(numeric_attrs, Diagnosis = data$Diagnosis)
numeric_long <- numeric_data %>%
  pivot_longer(-Diagnosis, names_to = "variable", values_to = "value")
ggplot(numeric_long, aes(x = value, fill = Diagnosis)) +
  geom_density(alpha = 0.5) +
  facet_wrap(~ variable, scales = "free", ncol = 3) +
  labs(title = "Density Plots of Numeric Variables by Diagnosis",
       x = "Value", y = "Density") +
  theme_minimal()

# Yet another one visualization, now density functions with respect to Diagnosis results


factor_data <- data %>%
  select_if(is.factor)

factor_data <- factor_data %>%
  dplyr::select(-EducationLevel, -Ethnicity)

factor_data %>%
  pivot_longer(-Diagnosis, names_to = "variable", values_to = "value") %>%
  group_by(Diagnosis, variable, value) %>%
  summarise(count = n(), .groups = "drop") %>%
  group_by(Diagnosis, variable) %>%
  mutate(percentage = count / sum(count)) %>%
  ggplot(aes(x = value, y = percentage, fill = Diagnosis)) +
  geom_bar(stat = "identity", position = "dodge") +
  facet_wrap(~ variable, scales = "free_x") +
  scale_y_continuous(labels = percent_format()) +
  labs(title = "Percentage Distribution by Diagnosis",
       x = "Category", y = "Percentage") +
  theme_minimal() +
  theme(axis.text.x = element_text(angle = 45, hjust = 1))

# Here we can see ratios/proportions for categoricals 


corr_mat <- cor(data_numeric, use = "pairwise.complete.obs")
corr_long <- as.data.frame(corr_mat) %>% 
  rownames_to_column("Var1") %>% 
  pivot_longer(-Var1, names_to = "Var2", values_to = "r")
ggplot(corr_long, aes(Var1, Var2, fill = r)) +
  geom_tile(color = "white") +
  geom_text(aes(label = sprintf("%.2f", r)), size = 2.5) +
  scale_fill_gradient2(
    low  = "lightblue",
    mid  = "white",
    high = "red",
    midpoint = 0, limits = c(-1, 1), name = "r"
  ) +
  labs(title = "Correlation heat-map – all variables incl. Alzheimer Diagnosis",
       x = NULL, y = NULL) +
  theme_minimal(base_size = 10) +
  theme(
    axis.text.x = element_text(angle = 45, hjust = 1),
    panel.grid  = element_blank()
  )
# From this plot we see that MMSE, ADL and FunctionalAssessment show the strongest (absolute) correlations with Diagnosis, followed by SleepQuality and CholesterolHDL. We therefore follow up withformal group-difference tests on all predictors.



# t-test for numeric predictors
t_tbl <- map_dfr(numeric_attrs_names, function(v){
  
  # Perform two-sample t-test comparing groups Diagnosis=0 vs Diagnosis=1 for variable v
  res <- t.test(data[[v]] ~ data$Diagnosis)
  
  # build a one-row tibble summarizing the t-test results for this variable
  tibble(
    Variable   = v, # name of the predictor
    Test       = "t‑test",# t-test
    
    # (difference in group means (AD - nonAD)) / (standard error of that difference)
    # SE is mean in difference of values if we took the same test many more times
    Statistic  = round(res$statistic, 3),
    p_value    = round(res$p.value , 5), # statistical significance
    Significant= ifelse(res$p.value < .05, "Yes", "No") # flag if p < 0.05
  )
})

# Chi-square test for categorical predictors
chi_tbl <- map_dfr(factor_attrs_names[factor_attrs_names!="Diagnosis"], function(v){
  
  # perform chi-square test on contingency table of v vs Diagnosis
  res <- chisq.test(table(data[[v]], data$Diagnosis))
  # build a one-row tibble summarizing the chi-square test results for this variable
  tibble(
    Variable   = v, # name of the predictor
    Test       = "χ²", # chi-square test
    
    # (difference in group means (AD - nonAD)) / (standard error of that difference)
    # SE is mean in difference of values if we took the same test many more times
    Statistic  = round(res$statistic, 3),
    p_value    = round(res$p.value , 5), # statistical significance
    Significant= ifelse(res$p.value < .05, "Yes", "No") # flag if p < 0.05
  )
})

# Combine both result tables & sort by p-value
test_results <- bind_rows(t_tbl, chi_tbl) %>% 
                arrange(p_value)
print(test_results)

predictors <- c(
  "FunctionalAssessment", "ADL", 
  "MemoryComplaints", "BehavioralProblems",
  "MMSE", "CholesterolHDL", "SleepQuality"
)


# Conclusion:
# -MMSE, ADL, MemoryComplaints,BehavioralProblems and FunctionalAssessment all show extremely low p-values (0), indicating highly significant group differences. They are strong numeric predictors.
# -SleepQuality (p = 0.0087) and CholesterolHDL (p = 0.049) semm to be also significant.
# -All other tests have p > 0.05, so no strong evidence.
# --MMSE, ADL, MemoryComplaints,BehavioralProblems and FunctionalAssessment, then consider SleepQuality & CholHDL in models as they also displayed p-values < 0.05.


# Visualizing the significant predictors by Diagnosis
# These boxplots and barplots illustrate the distribution of each variable in the No-AD (0) vs AD (1) groups, confirming the statistical differences we observed in the t-tests and chi-square tests.

ggplot(data, aes(x = Diagnosis, y = MMSE, fill = Diagnosis)) +
  geom_boxplot() +
  labs(title = "MMSE vs. Diagnosis", 
       x = "Diagnosis", 
       y = "MMSE value") +
  theme_minimal() +
  scale_x_discrete(labels = c("No Alzheimer's", "Alzheimer's"))

ggplot(data, aes(x = Diagnosis, y = ADL, fill = Diagnosis)) +
  geom_boxplot() +
  labs(title = "ADL vs. Diagnosis", 
       x = "Diagnosis", 
       y = "ADL value") +
  theme_minimal() +
  scale_x_discrete(labels = c("No Alzheimer's", "Alzheimer's"))

ggplot(data, aes(x = Diagnosis, y = FunctionalAssessment, fill = Diagnosis)) +
  geom_boxplot() +
  labs(title = "FunctionalAssessment vs. Diagnosis", 
       x = "Diagnosis", 
       y = "FunctionalAssessment value") +
  theme_minimal() +
  scale_x_discrete(labels = c("No Alzheimer's", "Alzheimer's"))

ggplot(data, aes(x = MemoryComplaints, fill = Diagnosis)) +
  geom_bar(position = "fill") +
  scale_x_discrete(labels = c("No complaint","Complaint")) +
  labs(title = "Memory complaints by AD status",
       x = "Memory complaints",
       y = "Count") +
  theme_minimal()

ggplot(data, aes(x = BehavioralProblems, fill = Diagnosis)) +
  geom_bar(position = "fill") +
  scale_x_discrete(labels = c("No problems","Problems")) +
  labs(title = "Behavioral problems by AD status",
       x = "Behavioral problems",
       y = "Count") +
  theme_minimal()

ggplot(data, aes(x = Diagnosis, y = CholesterolHDL, fill = Diagnosis)) +
  geom_boxplot() +
  labs(title = "CholesterolHDL vs. Diagnosis", 
       x = "Diagnosis", 
       y = "CholesterolHDL value") +
  theme_minimal() +
  scale_x_discrete(labels = c("No Alzheimer's", "Alzheimer's"))

ggplot(data, aes(x = Diagnosis, y = SleepQuality, fill = Diagnosis)) +
  geom_boxplot() +
  labs(title = "SleepQuality vs. Diagnosis", 
       x = "Diagnosis", 
       y = "SleepQuality value") +
  theme_minimal() +
  scale_x_discrete(labels = c("No Alzheimer's", "Alzheimer's"))


# EDA findings
# -No missing values in any column -> no imputation ( substituting missing data with a different value) needed
# -Slight diagnosis imbalance (65 % negative, 35 % positive) -> we will most likely adjust probability threshold to avoid bias or choose models which can work with this inequality
# -Numeric attributes - mostly uniform -> we keep all rows and skip outlier deletion
# -Categorical factors - most are highly imbalanced with the 0 level dominant
# –Correlation heatmap shows strongest associations with Diagnosis for MMSE, ADL, MemoryComplaints,BehavioralProblems and FunctionalAssessment, moderate for SleepQuality and HDL.
# –Categorical predictors mostly skewed to 0 level. Several (MemoryComplaints, BehavioralProblems) still show enough positives to use in models.
# Overall: the dataset is clean, and meets key modeling prerequisites. No further EDA-level fixes are required before moving to formal hypothesis creation/testing & model fitting.


################################################################################
############################ Hypothesis formulation ############################
################################################################################
# H0: The classification performance (e.g. AUC) of the base model built on MMSE, FunctionalAssessment, ADL, MemoryComplaints and BehavioralProblems is yealds little no none performance increase when we add the composite LRI. There is no gain from including the lifestyle index.

# H1: Adding the composite LRI to that core five-variable model produces a significant increase in classification performance (e.g. AUC).



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
















# Model Selection Rationale Based on Requirements and EDA:

# We have a binary response variable (Diagnosis) and a dataset with 2149 samples.
# Our EDA showed that no predictors are normally distributed, there's almost no correlation between predictors,
# and t-tests and chi-squared tests indicated significant associations between individual predictors and the response.
# There were no outliers and no missing data.

# --- Model Evaluation and Rejection Rationale ---

# 1. Logistic Regression (Chosen Model):
#    - Requirement: Binary outcome - MET (Diagnosis is binary).
#    - Requirement: Predictor Variable(s) Independence - ASSUMED (EDA showed low correlation).
#    - Requirement: No severe multicollinearity - MET (VIF ≤ 5, max 1.03 in previous table).
#    - Requirement: Normality - NOT REQUIRED.
#    - Requirement: Feature Scaling - NOT REQUIRED.
#    - Requirement: Homoscedasticity - NOT STRICTLY REQUIRED (though checked in residuals).
#    - Requirement: No extreme outliers - MET (EDA showed no outliers).
#    - Requirement: Large sample size - MET (2149 samples).
#    - Requirement: Linear relationship between predictors and the log-odds - ASSUMED and the model's functional form.
#    - Conclusion: Logistic Regression is a suitable choice given the binary outcome, lack of strong multicollinearity,
#                  and the fact that normality of predictors is not required.

# 2. Support Vector Machine (SVM):
#    - Requirement: Binary of multiclass - MET (can handle binary).
#    - Requirement: Predictor Variable(s) Independence - ASSUMED.
#    - Requirement: No severe multicollinearity - MET.
#    - Requirement: Normality - NOT REQUIRED.
#    - Requirement: Feature Scaling - RECOMMENDED (our data was likely not scaled, as it wasn't explicitly mentioned as done). While it *could* be performed, Logistic Regression doesn't strictly need it and was a good fit otherwise.
#    - Requirement: No extreme outliers - MET.
#    - Requirement: Performs well with smaller sample sizes, but larger sample size can boost performance - MET (large sample size).
#    - Requirement: Can handle non-linear relationships using kernel functions - POTENTIAL BENEFIT (EDA didn't strongly suggest non-linearity in predictors themselves, but in their relationship with the log-odds, Logistic Regression handles this through its link function).
#    - Rejection Rationale: While SVM could potentially work, Logistic Regression is often more interpretable for binary outcomes and directly models the probability. The lack of strong evidence for complex non-linear relationships in the predictors themselves didn't necessitate the complexity of SVM with kernels.

# 3. Linear Discriminant Analysis (LDA) / Quadratic Discriminant Analysis (QDA):
#    - Requirement: Categorical Response Variable - MET (binary is a type of categorical).
#    - Requirement: Predictor Variable(s) Independence - ASSUMED.
#    - Requirement: No severe multicollinearity - MET.
#    - Requirement: Predictor variables should be normally distributed - NOT MET (EDA showed no predictors are normally distributed). While LDA can be somewhat robust to mild violations, a complete lack of normality across all predictors is a significant issue.
#    - Requirement: Predictor variables should have the same variance (Homoscedasticity for LDA) - NOT CHECKED EXPLICITLY FOR PREDICTORS, but the lack of normality raises concerns. QDA relaxes this assumption but still prefers normality.
#    - Requirement: No extreme outliers - MET.
#    - Requirement: Performs better with larger sample sizes - MET.
#    - Requirement: Linear combinations of features should separate classes (LDA) - This might not hold if the underlying relationships are complex and non-linear in the predictor space.
#    - Rejection Rationale: The key reason for rejecting LDA/QDA is the violation of the normality assumption for the predictor variables. These methods rely heavily on the assumption that predictors within each class follow a multivariate normal distribution.











################################################################################
############################# 5. Data Spending #################################
################################################################################
# We pick one technique that: 
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








# Set up a consistent cross-validation strategy (e.g., 5-fold CV)
cv_consistent <- trainControl(method = "cv", number = 5)

# Choose a model (logistic regression)
model_type <- "glm"

# Train the model with the consistent CV strategy
set.seed(123)
model_5_fold <- train(Diagnosis ~ ., data = data[, c("Diagnosis", predictors)], method = model_type, trControl = cv_consistent, metric = "Accuracy")
cv_consistent <- trainControl(method = "cv", number = 10)

set.seed(123)
model_10_fold_SIMPLIFIED <- train(Diagnosis ~ ., data = data[, c("Diagnosis", predictors)], method = model_type, trControl = cv_consistent, metric = "Accuracy") # Using same CV

set.seed(123)
model_stratified_5_fold_SIMPLIFIED <- train(Diagnosis ~ ., data = data[, c("Diagnosis", predictors)], method = model_type, trControl = cv_consistent, metric = "Accuracy") # Using same CV

# LOOCV has a different number of resamples (equal to the number of data points),
# so it's best to exclude it from direct comparison with k-fold CV in resamples()
# model_loocv <- train(Diagnosis ~ ., data = data[, c("Diagnosis", predictors)], method = model_type, trControl = cv_consistent, metric = "Accuracy")

set.seed(123)
model_repeated_cv_SIMPLIFIED <- train(Diagnosis ~ ., data = data[, c("Diagnosis", predictors)], method = model_type, trControl = cv_consistent, metric = "Accuracy") # Using same CV

set.seed(123)
model_shuffle_split_SIMPLIFIED <- train(Diagnosis ~ ., data = data[, c("Diagnosis", predictors)], method = model_type, trControl = cv_consistent, metric = "Accuracy") # Using same CV

# Compare the results of models trained with the SAME number of resamples
results_comparison <- resamples(list(
  `5-Fold CV` = model_5_fold,
  `10-Fold CV (Simplified)` = model_10_fold_SIMPLIFIED,
  `Stratified 5-Fold CV (Simplified)` = model_stratified_5_fold_SIMPLIFIED,
  # `LOOCV` = model_loocv, # Excluding LOOCV for now due to different resample number
  `Repeated CV (Simplified)` = model_repeated_cv_SIMPLIFIED,
  `Shuffle-Split (Simplified)` = model_shuffle_split_SIMPLIFIED
))

summary(results_comparison)

# Visualize the performance distributions
dotplot(results_comparison, metric = "Accuracy")




################################################################################
################# Data preparation & Lifestyle-Risk-Index (LRI) ################
################################################################################
# Goal is to keep only variables needed for the hypothesis + create a trimmed data frame for modelling

# Keep only variables required by the hypothesis
core_vars <- c(predictors, 
  # non-modifiable risk factors
  "FamilyHistoryAlzheimers", "HeadInjury",
  # lifestyle components
  "PhysicalActivity", "DietQuality",
  "Smoking", "AlcoholConsumption", "BMI"
)

data_hyp <- data_numeric %>% dplyr::select(all_of(core_vars))

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
#str(data_hyp)

data_numeric <- data_numeric %>% mutate(
  LRI = data_hyp$LRI
)
data <- data %>% mutate(
  LRI = data_hyp$LRI
)
#str(data)


# Final fix of factor levels & drop rows with any missing values
#data_hyp <- data_hyp %>% 
#  mutate(
#    Diagnosis = factor(Diagnosis, levels = c(0, 1)),  # ensure 0 = No, 1 = Yes
#    Gender    = factor(Gender)                        # guarantee factor type
#  ) %>% 
#  drop_na()









################################################################################
################################ Hypothesis test ###############################
################################################################################
# H1: Higher LRI = higher odds of Alzheimer.


# Split dataset into training and testing sets
set.seed(235455)
train_index <- createDataPartition(data_numeric$Diagnosis, p = 0.7, list = FALSE)
train_data <- data_numeric[train_index, ]
test_data  <- data_numeric[-train_index, ]

model_primary <- glm(
  Diagnosis ~ .,
  data = train_data[, c("Diagnosis", predictors)],
  family = binomial
)

summary(model_primary)

second_predictors <- c(predictors, "LRI")
model_secondary <- glm(
  Diagnosis ~ .,
  data = train_data[, c("Diagnosis", second_predictors[!second_predictors %in% c("CholesterolHDL", "SleepQuality")])],
  family = binomial
)
# SleepQuality is also in LRI predictor

summary(model_secondary)


# Compare models
AIC(model_primary, model_secondary)
# AIC index shows that priamry/first model is slightly better, so we continue with primary model.

# Multicolinearity
vif(model_primary)
# VIF values are < 5 therefore no multicolinearity is detected.

# TODO: keep this ? know how to explain it ?
# Convert log-odds to odds-ratios
exp(cbind(OR = coef(model_primary),   # exponentiate → odds ratios
          confint(model_primary)))    # 95 % CIs for interpretation

library(leaps)
combination_models <- regsubsets(Diagnosis ~ .,
                     data = train_data[, c("Diagnosis", second_predictors)])
summary(combination_models)
# This resulting tables shows best combination of predictors for N number of
# predictors e.g. if we want just 1 - we pick FunctionalAssessment, for 2 predictors
# we pick FunctionalAssessment and MemoryComplaints, and for 3 we pick
# FunctionalAssessment, MemoryComplaints and ADL, ...








# Checking for the 3 Conditions

# 1. Randomness (Visual Check from QQ-Plot):
#    - If the points generally follow the red line, it suggests the residuals are
#      not systematically deviating from the expected distribution under the model.
#    - Large deviations, especially curves or S-shapes, might indicate a lack of randomness
#      or model misspecification.

# 2. Outliers (Visual Check from QQ-Plot):
#    - Points that are far away from the red line, particularly at the extreme ends (tails)
#      of the plot, can be potential outliers. These are observations that are much more
#      or less extreme than expected by the model.

# 3. Heteroscedasticity (Less Directly Assessed by QQ-Plot):
#    - A QQ-plot primarily assesses the distribution of the residuals, not directly their
#      variance across predicted values.
#    - However, severe deviations from the line might indirectly hint at issues with the
#      variance if the model assumptions are violated.
#    - To more directly assess heteroscedasticity, the "Residuals vs. Predicted Values" plot
#      (as shown in the previous response) is more informative. Look for a funnel shape
#      or increasing/decreasing spread of residuals as predicted probabilities change.

residuals_primary <- residuals(model_primary, type = "deviance")
#plot(model_primary)


# --- QQ-Plot for Residuals ---
#qqnorm(residuals_primary, main = "QQ-Plot of Deviance Residuals (Primary Model)")
#qqline(residuals_primary, col = "red")

car::qqPlot(residuals_primary, main = "Q-Q Plot of Residuals from Linear Regression")

# Interpretation of QQ-Plot:
# This plot helps assess if the distribution of your residuals follows a theoretical normal distribution.
# - If the points fall roughly along the red line, the residuals are approximately normally distributed.
# - Deviations from the line, especially S-shapes or curves, suggest non-normality.
# - Points far away from the line at the ends indicate potential outliers (residuals that are much larger or smaller than expected).
# Note: For logistic regression, perfect normality of residuals is not expected due to the binary outcome. Focus on severe deviations.

# --- Statistical Tests on Residuals ---

# Perform tests
shapiro_test_result <- shapiro.test(residuals_primary)
ttest_result <- t.test(residuals_primary)
bp_test_result <- bptest(model_primary)
tpt_result <- turning.point.test(residuals_primary)
dwtest(model_primary)


# --- Table of Test Results ---

test_results_table <- data.frame(
  Test = c("Shapiro-Wilk Normality Test",
           "One-Sample t-test (Mean of Residuals)",
           "Breusch-Pagan Test for Homoscedasticity",
           "Turning Point Test for Randomness",
           "Durbin-Watson Test for Autocorrelation (Caution: GLM)"),
  Statistic = c(round(shapiro_test_result$statistic, 3),
                round(ttest_result$statistic, 3),
                round(bp_test_result$statistic, 3),
                round(tpt_result$statistic, 3),
                round(dw_test_result$statistic, 3)),
  P_value = c(format.pval(shapiro_test_result$p.value, digits = 3),
              format.pval(ttest_result$p.value, digits = 3),
              format.pval(bp_test_result$p.value, digits = 3),
              format.pval(tpt_result$p.value, digits = 3),
              format.pval(dw_test_result$p.value, digits = 3)),
  Interpretation = character(5) # Placeholder for GOOD/BAD
)

# Fill the Interpretation column with GOOD/BAD
test_results_table$Interpretation[1] <- ifelse(shapiro_test_result$p.value > 0.05, "GOOD", "BAD")
test_results_table$Interpretation[2] <- ifelse(ttest_result$p.value > 0.05, "GOOD", "BAD")
test_results_table$Interpretation[3] <- ifelse(bp_test_result$p.value > 0.05, "GOOD", "BAD")
test_results_table$Interpretation[4] <- ifelse(tpt_result$p.value > 0.05, "GOOD", "BAD")
test_results_table$Interpretation[5] <- ifelse(is.na(dw_test_result$p.value), "N/A", ifelse(dw_test_result$p.value > 0.05, "GOOD", "BAD"))

print(test_results_table)

# --- Interpretation by Outcome ---

outcome_interpretation <- data.frame(
  Test = test_results_table$Test,
  Outcome = character(5)
)

outcome_interpretation$Outcome[1] <- ifelse(shapiro_test_result$p.value > 0.05,
                                            "Residuals approximately normally distributed (for GLM, not strictly required).",
                                            "Residuals deviate significantly from normality (common in GLMs).")
outcome_interpretation$Outcome[2] <- ifelse(ttest_result$p.value > 0.05,
                                            "Mean of residuals is close to zero (suggests unbiased model).",
                                            "Mean of residuals is significantly different from zero (potential model bias).")
outcome_interpretation$Outcome[3] <- ifelse(bp_test_result$p.value > 0.05,
                                            "Variance of residuals is approximately constant (homoscedasticity assumed).",
                                            "Variance of residuals is not constant (heteroscedasticity detected).")
outcome_interpretation$Outcome[4] <- ifelse(tpt_result$p.value > 0.05,
                                            "Residuals appear to be randomly distributed in sequence.",
                                            "Residuals show evidence of non-random patterns in sequence (potential unmodeled dependencies).")
outcome_interpretation$Outcome[5] <- ifelse(is.na(dw_test_result$p.value),
                                            "Durbin-Watson test result not available.",
                                            ifelse(dw_test_result$p.value > 0.05,
                                                   "Residuals appear to be independent (no significant autocorrelation).",
                                                   "Residuals show significant autocorrelation (potential unmodeled temporal dependencies)."))

cat("\n--- Interpretation of Residual Tests by Outcome ---\n")
print(outcome_interpretation)








# Predict probabilities on test set
test_probs <- predict(model_primary, newdata = test_data, type = "response")

# ROC analysis
roc_obj_primary <- roc(test_data$Diagnosis, test_probs)
auc_primary <- auc(roc_obj_primary)
cat("AUC:", round(auc_primary, 3), "\n")

# Step 6: Get optimal threshold (Youden's J)
coords_obj_primary <- coords(roc_obj_primary, "best", best.method = "youden", ret = c("threshold", "sensitivity", "specificity"))
optimal_threshold_primary <- coords_obj_primary["threshold"]
optimal_threshold_primary

# Step 7: Confusion matrix - threshold 0.5
preds_default <- factor(ifelse(test_probs > 0.5, 1, 0), levels = c(0, 1))
conf_matrix_default <- confusionMatrix(preds_default, as.factor(test_data$Diagnosis), positive = "1")
cat("\nConfusion Matrix (Threshold 0.5):\n")
print(conf_matrix_default)

optimal_threshold_primary
# Confusion matrix - optimal threshold
preds_optimal <- factor(ifelse(test_probs > 0.3338061, 1, 0), levels = c(0, 1))
conf_matrix_optimal <- confusionMatrix(preds_optimal, as.factor(test_data$Diagnosis), positive = "1")
cat("\nConfusion Matrix (Optimal Threshold):\n")
print(conf_matrix_optimal)

# --- Visualize Confusion Matrices
create_cm_df <- function(cm, model_name) {
  as.data.frame(cm$table) %>%
    mutate(Model = model_name)
}

cm_df_primary_default <- create_cm_df(conf_matrix_default, "Primary (Th=0.5)")
cm_df_primary_optimal <- create_cm_df(conf_matrix_optimal, "Primary (Th=0.4381873)")

cm_df_primary <- rbind(cm_df_primary_default, cm_df_primary_optimal)

ggplot(cm_df_primary, aes(x = Reference, y = Prediction, fill = Freq)) +
  geom_tile() +
  geom_text(aes(label = Freq), color = "white", size = 5) +
  scale_fill_gradient(low = "lightblue", high = "steelblue") +
  facet_wrap(~ Model) +
  labs(title = "Confusion Matrices - Primary Model", fill = "Count") +
  theme_minimal() +
  theme(aspect.ratio = 1)




# Predict probabilities on test set for secondary model
test_probs_secondary <- predict(model_secondary, newdata = test_data, type = "response")

# ROC analysis for secondary model
roc_obj_secondary <- roc(test_data$Diagnosis, test_probs_secondary)
auc_secondary <- auc(roc_obj_secondary)
cat("AUC (Secondary Model):", round(auc_primary, 3), "\n")
cat("AUC (Secondary Model):", round(auc_secondary, 3), "\n")

# Here secondary model has slightly bigger auc.

# Optimal threshold (Youden's J) for secondary model
coords_obj_secondary <- coords(roc_obj_secondary, "best", best.method = "youden", ret = c("threshold", "sensitivity", "specificity"))
optimal_threshold_secondary <- coords_obj_secondary["threshold"]
optimal_threshold_secondary

# --- Visualize ROC Curves ---
roc_df_primary <- data.frame(
  fpr = 1 - roc_obj_primary$specificities,
  tpr = roc_obj_primary$sensitivities,
  model = "Primary Model"
)

roc_df_secondary <- data.frame(
  fpr = 1 - roc_obj_secondary$specificities,
  tpr = roc_obj_secondary$sensitivities,
  model = "Secondary Model"
)

roc_df_combined <- rbind(roc_df_primary, roc_df_secondary)



ggplot(roc_df_combined, aes(x = fpr, y = tpr, color = model)) +
  geom_line(linewidth = 1) +
  geom_abline(intercept = 0, slope = 1, linetype = "dashed", color = "grey") +
  
  # Add vertical and horizontal lines for primary threshold
  geom_vline(xintercept = 1 - coords_obj_primary$specificity, color = "blue", linetype = "dotted") +
  geom_hline(yintercept = coords_obj_primary$sensitivity, color = "blue", linetype = "dotted") +
  
  # Add vertical and horizontal lines for secondary threshold
  geom_vline(xintercept = 1 - coords_obj_secondary$specificity, color = "red", linetype = "dotted") +
  geom_hline(yintercept = coords_obj_secondary$sensitivity, color = "red", linetype = "dotted") +
  
  labs(
    title = "Comparison of ROC Curves",
    x = "False Positive Rate (1 - Specificity)",
    y = "True Positive Rate (Sensitivity)",
    color = "Model"
  ) +
  theme_minimal() +
  theme(legend.position = "bottom") +
  annotate("text", x = 0.75, y = 0.25, label = paste("AUC (Primary) =", round(auc_primary, 3)), color = "blue") +
  annotate("text", x = 0.75, y = 0.20, label = paste("AUC (Secondary) =", round(auc_secondary, 3)), color = "red")



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






# articles:

# https://ieeexplore.ieee.org/abstract/document/9521165
# https://ieeexplore.ieee.org/abstract/document/10359516
# https://ieeexplore.ieee.org/abstract/document/10353401

# https://ieeexplore.ieee.org/abstract/document/9201455
# https://www.sciencedirect.com/science/article/pii/S1319157820304936
# https://www.sciencedirect.com/science/article/pii/S0169809522002241

# https://link.springer.com/article/10.1186/s12911-023-02238-9




# https://ieeexplore.ieee.org/abstract/document/9201455 **************
# https://www.sciencedirect.com/science/article/pii/S1319157820304936 ************




gbm_model <- gbm(
  Diagnosis ~ ., # Formula: Predict 'Diagnosis' using all other columns in the specified data
  data = train_data[, c("Diagnosis", predictors)], # Data frame and the columns to be used for training (Diagnosis and the selected predictors)
  distribution = "bernoulli", # - logistic # Specifies the loss function to be used. "bernoulli" is appropriate for binary classification (0 or 1 outcomes like Diagnosis)
  n.trees = 100, # The number of boosting iterations to perform. Each tree is built sequentially, trying to correct the errors of the previous trees. More trees can lead to better performance but also increased risk of overfitting and longer training time.
  shrinkage = 0.1, # Also known as the learning rate. It scales the contribution of each tree. A smaller shrinkage value (e.g., 0.01, 0.1) makes the learning slower but often results in more robust models that generalize better.
  interaction.depth = 6, # The maximum depth of each tree. It controls the complexity of the interactions between predictors that the model can capture. A depth of 1 means the model is additive (no interactions), a depth of 2 allows for two-way interactions, and so on. Higher values can capture more complex relationships but also increase the risk of overfitting.
  n.minobsinnode = 10, # The minimum number of observations in the terminal nodes (leaves) of each tree. This parameter helps to regularize the model and prevent overfitting by ensuring that leaves are not too specific to small subsets of the training data.
  verbose = FALSE # A logical value indicating whether to print progress and performance updates during the training process. FALSE means no output during training.
)

# Check summary (variable importance)
summary(gbm_model)

# Predict probabilities
pred_probs <- predict(gbm_model, newdata = test_data, n.trees = 100, type = "response")

# Convert probabilities to class labels (0 or 1) using 0.5 threshold
pred_labels <- ifelse(pred_probs > 0.5, 1, 0)

# Actual labels
true_labels <- test_data$Diagnosis

# Confusion matrix
conf_mat <- confusionMatrix(as.factor(pred_labels), as.factor(true_labels), positive = "1")
print(conf_mat)

cm_df <- as.data.frame(conf_mat$table)

# Rename columns for clarity
cm_df <- cm_df %>%
  rename(Predicted = Prediction, Actual = Reference, Count = Freq)

# Create the plot
ggplot(cm_df, aes(x = Actual, y = Predicted, fill = Count)) +
  geom_tile() +
  geom_text(aes(label = Count), color = "white", size = 5) +
  scale_fill_gradient(low = "lightblue", high = "steelblue") +
  labs(title = "Confusion Matrix", fill = "Count") +
  theme_minimal() +
  theme(aspect.ratio = 1)


# AUC
roc_obj_gboost <- roc(true_labels, pred_probs)
auc_gboost <- auc(roc_obj_gboost)
cat("AUC:", auc_gboost, "\n")

# Accuracy reached 95,76%


coords_obj_gboost <- coords(roc_obj_gboost, "best", best.method = "youden", ret = c("threshold", "sensitivity", "specificity"))
optimal_threshold_gboost <- coords_obj_gboost["threshold"]
optimal_threshold_gboost


preds_optimal <- factor(ifelse(pred_probs > 0.3567714, 1, 0), levels = c(0, 1))
conf_matrix_optimal <- confusionMatrix(preds_optimal, as.factor(test_data$Diagnosis), positive = "1")
cat("\nConfusion Matrix (Optimal Threshold):\n")
print(conf_matrix_optimal)



cm_df_primary_default <- create_cm_df(conf_mat, "Primary (Th=0.5)")
cm_df_primary_optimal <- create_cm_df(conf_matrix_optimal, "Primary (Th=0.3567714)")

cm_df_primary <- rbind(cm_df_primary_default, cm_df_primary_optimal)

ggplot(cm_df_primary, aes(x = Reference, y = Prediction, fill = Freq)) +
  geom_tile() +
  geom_text(aes(label = Freq), color = "white", size = 5) +
  scale_fill_gradient(low = "lightblue", high = "steelblue") +
  facet_wrap(~ Model) +
  labs(title = "Confusion Matrices - Primary Model", fill = "Count") +
  theme_minimal() +
  theme(aspect.ratio = 1)



# --- Visualize ROC Curves ---
roc_df_primary <- data.frame(
  fpr = 1 - roc_obj_primary$specificities,
  tpr = roc_obj_primary$sensitivities,
  model = "Primary Model"
)

roc_df_gboost <- data.frame(
  fpr = 1 - roc_obj_gboost$specificities,
  tpr = roc_obj_gboost$sensitivities,
  model = "GBoost Model"
)

roc_df_combined <- rbind(roc_df_primary, roc_df_gboost)

ggplot(roc_df_combined, aes(x = fpr, y = tpr, color = model)) +
  geom_line(linewidth = 1) +
  geom_abline(intercept = 0, slope = 1, linetype = "dashed", color = "grey") +
  
  labs(
    title = "Comparison of ROC Curves",
    x = "False Positive Rate (1 - Specificity)",
    y = "True Positive Rate (Sensitivity)",
    color = "Model"
  ) +
  theme_minimal() +
  theme(legend.position = "bottom") +
  annotate("text", x = 0.75, y = 0.25, label = paste("AUC (Primary) =", round(auc_primary, 3)), color = "blue") +
  annotate("text", x = 0.75, y = 0.20, label = paste("AUC (GBoost) =", round(auc_gboost, 3)), color = "red")


# Gradient Boosting Machine (GBM) Model Results:

# 1. Variable Importance:
#    The summary of the GBM model indicates the relative influence of each predictor variable on the model's predictions.
#    - FunctionalAssessment is the most important variable, with a relative influence of approximately 23.9%.
#    - MMSE and ADL are also highly influential, with relative influences of around 21.7% and 20.7%, respectively.
#    - MemoryComplaints and BehavioralProblems have moderate influence (around 15.7% and 13.3%).
#    - SleepQuality and CholesterolHDL appear to have the least influence on the model's predictions (2.7% and 1.9%).

# 2. Confusion Matrix and Statistics:
#    The confusion matrix on the test set (assuming this matrix was generated from the GBM model's predictions) shows the following:
#    - True Negatives (TN): 386 (Number of actual 0s correctly predicted as 0)
#    - False Negatives (FN): 25 (Number of actual 1s incorrectly predicted as 0)
#    - False Positives (FP): 13 (Number of actual 0s incorrectly predicted as 1)
#    - True Positives (TP): 220 (Number of actual 1s correctly predicted as 1)

#    Performance Metrics:
#    - Accuracy: 0.941 (94.1% of predictions were correct)
#    - 95% Confidence Interval (CI) for Accuracy: (0.9199, 0.9579)
#    - No Information Rate (NIR): 0.6196 (Accuracy if the model always predicted the majority class)
#    - P-Value [Acc > NIR]: < 2e-16 (Highly significant, indicating the model's accuracy is significantly better than random guessing)
#    - Kappa: 0.8736 (Measure of agreement between predicted and actual classifications, with 1 being perfect agreement)
#    - Mcnemar's Test P-Value: 0.07435 (P-value for McNemar's chi-squared test, testing for symmetry of errors. A value > 0.05 suggests no significant difference in the types of errors)
#    - Sensitivity (Recall, True Positive Rate): 0.8980 (Proportion of actual 1s correctly identified)
#    - Specificity (True Negative Rate): 0.9674 (Proportion of actual 0s correctly identified)
#    - Positive Predictive Value (Precision): 0.9442 (Proportion of predicted 1s that were actually 1s)
#    - Negative Predictive Value: 0.9392 (Proportion of predicted 0s that were actually 0s)
#    - Prevalence: 0.3804 (Proportion of the positive class (1) in the test set)
#    - Detection Rate: 0.3416 (Proportion of the positive class correctly identified out of the total number of instances)
#    - Detection Prevalence: 0.3618 (Proportion of instances predicted as positive)
#    - Balanced Accuracy: 0.9327 (Average of sensitivity and specificity)
#    - 'Positive' Class: 1 (Indicates that class '1' was treated as the positive class for sensitivity and specificity calculations)

# 3. Area Under the ROC Curve (AUC):
#    - AUC: 0.9534141 (This indicates excellent discriminatory power of the GBM model. An AUC of 0.95 suggests a high ability to distinguish between the positive and negative classes.)

# Findings:
# - The GBM model demonstrates strong predictive performance for the 'Diagnosis' variable, achieving a high accuracy and AUC.
# - FunctionalAssessment, MMSE, and ADL are the most important predictors in the model.
# - The model shows a good balance between sensitivity and specificity, indicating it performs well in identifying both positive and negative cases.
# - The high AUC further supports the model's ability to effectively separate the two diagnostic classes.