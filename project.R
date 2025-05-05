#' ---
#' title: "RMD_Oznal_Melikant_Kello"
#' author: "Adam Melikant, Richard Kello"
#' date: "`r Sys.Date()`"
#' output: html_document
#' ---


#'# Setup & data import
library(tidyverse)
library(tidymodels)
library(magrittr)
library(caret)
library(pROC)
library(corrplot)
library(ggmosaic)
library(dplyr)
library(MASS)
library(car)
library(latexpdf)
library(rsample)  
library(purrr)
library(gbm)
library(randtests)
library(lmtest)
library(parsnip)
library(workflows)
library(yardstick)
#'library(shiny)

data <- read.csv("alzheimers_disease_data.csv")



#'# Initial Data Exploration

#' Drop pointless columns like the DoctorInCharge since it is empty and no use to us and PatientID.
data <- data %>% dplyr::select(-DoctorInCharge, -PatientID)
data_numeric <- data
#'Manual look at data
str(data)     # general info about columns (type...)
summary(data) # info about each column (min, max, median...)

sum(is.na(data)) # how many are NAs in ds
colSums(is.na(data)) # how many are NAs in each column

#' Check for duplicated rows
any( duplicated(data) )

#' Check outcome balance
table(data$Diagnosis)             # AD diagnosis; 0 - No, 1 - Yes
prop.table(table(data$Diagnosis)) # AD diagnosis by percentage
#' Dataset has more negatives diagnosis than positive (65%)

#' Convert low-cardinality (few unique values) numerics to factors
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

#' Normality
normality_p_values <- apply(numeric_attrs, 2, function(x) {
  shapiro.test(x)$p.value
})
print(sort(normality_p_values, decreasing = T))
#' P-values < 0.05 suggest a deviation from normality. Consider transformations if needed.
#' we can observe that none or numeric attributes in normally distributed.


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

#' Yet another one visualization, now density functions with respect to Diagnosis results


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

#' Here we can see ratios/proportions for categoricals 


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
#' From this plot we see that MMSE, ADL and FunctionalAssessment show the strongest (absolute) correlations with Diagnosis, followed by SleepQuality and CholesterolHDL. We therefore follow up withformal group-difference tests on all predictors.



#' t-test for numeric predictors
t_tbl <- map_dfr(numeric_attrs_names, function(v){
  
  # Perform two-sample t-test comparing groups Diagnosis=0 vs Diagnosis=1 for variable v
  res <- t.test(data[[v]] ~ data$Diagnosis)
  
  # build a one-row tibble summarizing the t-test results for this variable
  tibble(
    Variable   = v, #' name of the predictor
    Test       = "t‑test",#' t-test
    
    # (difference in group means (AD - nonAD)) / (standard error of that difference)
    # SE is mean in difference of values if we took the same test many more times
    Statistic  = round(res$statistic, 3),
    p_value    = round(res$p.value , 5), #' statistical significance
    Significant= ifelse(res$p.value < .05, "Yes", "No") #' flag if p < 0.05
  )
})

#' Chi-square test for categorical predictors
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

#' Combine both result tables & sort by p-value
test_results <- bind_rows(t_tbl, chi_tbl) %>% 
  arrange(p_value)
print(test_results)

predictors <- c(
  "FunctionalAssessment", "ADL", 
  "MemoryComplaints", "BehavioralProblems",
  "MMSE", "CholesterolHDL", "SleepQuality", 
  
  "Ethnicity", "Hypertension", "CholesterolLDL",
  
  "FamilyHistoryAlzheimers", "Disorientation", "Forgetfulness"
)


#' Conclusion:
#' -MMSE, ADL, MemoryComplaints,BehavioralProblems and FunctionalAssessment all show extremely low p-values (0), indicating highly significant group differences. They are strong numeric predictors.
#' -SleepQuality (p = 0.0087) and CholesterolHDL (p = 0.049) semm to be also significant.
#' -All other tests have p > 0.05, so no strong evidence.
#' --MMSE, ADL, MemoryComplaints,BehavioralProblems and FunctionalAssessment, then consider SleepQuality & CholHDL in models as they also displayed p-values < 0.05.


#' Visualizing the significant predictors by Diagnosis
#' These boxplots and barplots illustrate the distribution of each variable in the No-AD (0) vs AD (1) groups, confirming the statistical differences we observed in the t-tests and chi-square tests.

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


#' EDA findings
#' -No missing values in any column -> no imputation ( substituting missing data with a different value) needed
#' -Slight diagnosis imbalance (65 % negative, 35 % positive) -> we will most likely adjust probability threshold to avoid bias or choose models which can work with this inequality
#' -Numeric attributes - mostly uniform -> we keep all rows and skip outlier deletion
#' -Categorical factors - most are highly imbalanced with the 0 level dominant
#' –Correlation heatmap shows strongest associations with Diagnosis for MMSE, ADL, MemoryComplaints,BehavioralProblems and FunctionalAssessment, moderate for SleepQuality and HDL.
#' –Categorical predictors mostly skewed to 0 level. Several (MemoryComplaints, BehavioralProblems) still show enough positives to use in models.
#' Overall: the dataset is clean, and meets key modeling prerequisites. No further EDA-level fixes are required before moving to formal hypothesis creation/testing & model fitting.


#'# Hypothesis formulation
#' H1: Age, Gender, FamilyHistoryAlzheimers, HeadInjury, MMSE, FunctionalAssessment, ADL,
#'     MemoryComplaints, BehavioralProblems, CholesterolHDL, SleepQuality, EducationLevel,
#'     and Ethnicity are all significantly associated with Alzheimer’s Disease diagnosis.



#' Based on EDA we selected 5 core predictors (MMSE, ADL, MemoryComplaints, BehavioralProblems, FunctionalAssessment) and concluded our models must check these requirements:
#' -Handle a binary outcome (Diagnosis ∈ {0,1})
#' -Accept a mix of numeric & factor predictors
#' -Work with 0% missing data (we have no NAs)
#' -Handle class imbalance (65% non-AD / 35% AD)

#' Algorithms selected:
#' 1) Logistic Regression (GLM)
#'    -Models log-odds of AD as a weighted sum of predictors.
#'    -Each coefficient shows how much that feature shifts AD risk.
#'    -Works well on our 65/35 class split. 
#'    -We’ll pick the best probability cutoff via cross-validation.
#' 2) Gradient Boosting Machine (GBM)
#'    -Builds many small decision trees in sequence to improve predictions.
#'    -Automatically handles both numeric and factor inputs without extra preprocessing.
#'    -Captures non-linear effects and feature interactions that a straight-line model might miss.
   
#’ Other methods considered (not fitted):
#’ 3) Linear Discriminant Analysis (LDA)
#’    – Assumes multivariate normality of the numeric predictors in each class.
#’    – Requires all inputs to be numeric and homoscedastic.
#’    – Dropped because we have mixed factor/numeric features and normality assumptions are violated.
#’ 4) Support Vector Machine (SVM)
#’    – Can model non-linear decision boundaries via kernel functions.
#’    – Requires careful kernel/parameter tuning and scaling. Lower interpretability.
#’    – Dropped due to lack of evidence for complex non-linear structure in EDA and our priority on model transparency.

#' We will therefore fit:
#' -Primary_model <- Logistic GLM
#' -Exploratory_model <- GBM



# We picked Stratified 10-fold cross-validation:
# -The data is divided into 10 equal “folds” of ~215 observations each.
# -In each of 10 iterations:
# --9 folds (≈1,934 samples) are used to train the model.
# --The remaining 1 fold (≈215 samples) is held out to test it.
# -Because we set strata = "Diagnosis", each fold preserves the 65/35 AD/non-AD ratio.
# -Over all folds, every sample is used exactly once for testing and nine times for training.
# -Averaging metrics (e.g. AUC, accuracy) across folds gives a robust estimate of expected performance and its variability.

#’ Alternative resampling schemes (besides Stratified 10-fold CV):
#’ -Stratified repeated k-fold CV (e.g. 5×10-fold):  
#’ --Like 10-fold CV, but runs it multiple times with new fold splits  
#’ --Further reduces variance of performance estimates  
#’ -Stratified Monte Carlo CV (repeated random splits):  
#’ --Randomly split into, say, 70/30 train/test many times  
#’ --Flexible train/test ratio, but may leave out some samples consistently  
#’ -Stratified bootstrap:  
#’ --Draw many bootstrap samples, train on each, test on out-of-bag cases  
#’ --Gives confidence intervals, but can be optimistically biased  
#’ -Leave-One-Out CV (LOOCV):  
#’ --Train on all but one sample each round (2,148/1)  
#’ --Maximizes training data but yields high variance and long runtimes  

#’ Why 10-fold CV?  
#’ -Balanced bias/variance: more stable than a single split, lighter than LOOCV  
#’ -Preserves 65/35 class ratio in every fold  
#’ -Uses all 2,149 cases efficiently for both training and testing  
#’ -Computationally cheap (10 model fits) yet standard for binary classifiers



#' Build cv_splits (stratified 10-fold CV)
set.seed(65778275)
cv_splits <- rsample::vfold_cv(
  data,
  v      = 10, # 10 folds
  strata = "Diagnosis" # keep 65/35 AD vs non-AD ratio in each fold
)
print(cv_splits)


#' Use those splits to estimate CV performance (no new data is created here. fit_resamples will train on 9/10 folds and test on the held-out 1/10 in each round)
#' Specify a logistic regression model
glm_spec <- logistic_reg() %>% set_engine("glm")  

#' Build a workflow that bundles the model and the formula together
wf <- workflow() %>%
  add_model(glm_spec) %>% # attach the logistic regression spec
  add_formula(
    Diagnosis ~ FunctionalAssessment + ADL +
      MemoryComplaints + BehavioralProblems +
      MMSE + SleepQuality + CholesterolHDL
  ) # define the predictors to use in one step

#' Fit the workflow across each CV split
glm_res <- fit_resamples(
  wf,
  resamples = cv_splits,
  metrics    = metric_set(roc_auc, accuracy), # compute AUC and accuracy on each held-out fold
  control    = control_resamples(save_pred = TRUE) # keep the per-fold predictions for diagnostics
)

#' Display results of each metric across the 10 folds
collect_metrics(glm_res)
#' We see:
#' -Mean accuracy = 0.846 (+/-0.008 -> std_err): on average, the model correctly classifies 84.6% of cases
#' -Mean ROC AUC  = 0.903 (+/-0.005 -> std_err): strong discrimination between AD vs. non-AD
#' The low standard errors across the 10 folds indicate that performance is stable and not overly sensitive to any single split.


#'# Model Fiting

#' Split dataset into training and testing sets
set.seed(65778275)
train_index <- createDataPartition(data$Diagnosis, p = 0.7, list = FALSE)
train_data  <- data[train_index, ]
test_data   <- data[-train_index, ]

#' # Fit the primary logistic regression model on the training data
model_primary <- glm(
  Diagnosis ~ FunctionalAssessment + ADL +
    MemoryComplaints + BehavioralProblems +
    MMSE + SleepQuality + CholesterolHDL,
  data   = train_data,
  family = binomial
)

summary(model_primary)
#'  Coefficients:
#'  –FunctionalAssessment: higher scores make AD less likely  
#'  –ADL: higher daily-living scores make AD less likely  
#'  –MemoryComplaints & BehavioralProblems: if present, these strongly increase AD risk  
#'  –MMSE: better cognition scores make AD less likely  
#'  –SleepQuality: weak evidence for protection (p≈0.05)  
#'  –HDL cholesterol: small increase in AD risk per unit rise  
#'  In other words, all predictors except SleepQuality are clearly significant at p<0.05.

#' Evaluate that final GLM on the held-out 30%
test_probs <- predict(model_primary, test_data, type = "response")
test_pred_class <- factor(ifelse(test_probs > 0.5, "1", "0"),
                          levels = c("0","1"))

#' Build a tibble with exactly two columns for roc_auc to handle factor
df_pred <- tibble(
  truth   = factor(test_data$Diagnosis, levels = c(0, 1)),
  .pred_1 = test_probs
)

conf   <- confusionMatrix(test_pred_class, factor(test_data$Diagnosis))
res_auc <- df_pred %>%
  roc_auc(truth = truth, .pred_1, event_level = "second")

print(conf)
#' Accuracy = 0.8339: model correctly classifies 83.4% of cases at 0.5 cutoff  
#' No Information Rate = 0.646: always predicting “no AD” for every patient, which is 64.6% of the data
#' Sensitivity = 0.8870: correctly identifies 88.7% of AD cases  
#' Specificity = 0.7368: correctly identifies 73.7% of non-AD cases  
#' Balanced Acc. = 0.8119: average of sensitivity & specificity  
#' 
print(res_auc)
#' AUC:  
#' This nearly matches the CV AUC (~0.90), confirming strong discrimination


library(leaps)
combination_models <- regsubsets(Diagnosis ~ .,
                                 data = train_data[, c("Diagnosis", predictors)])
summary(combination_models)
#' This resulting tables shows best combination of predictors for N number of
#' predictors e.g. if we want just 1 - we pick FunctionalAssessment, for 2 predictors
#' we pick FunctionalAssessment and ADL, and for 3 we pick
#' FunctionalAssessment, MemoryComplaints and ADL, ...

#' so we pick best combination of predictors
predictors <- c(
  "FunctionalAssessment", "ADL", 
  "MemoryComplaints", "BehavioralProblems",
  "MMSE", "SleepQuality", 
  "CholesterolLDL", "FamilyHistoryAlzheimers"
)

model_primary <- glm(
  Diagnosis ~ .,
  data = train_data[, c("Diagnosis", predictors)],
  family = binomial
)

summary(model_primary)
#' # Final Logistic Regression Results
#' We fitted a logistic model to predict Alzheimer’s diagnosis using our selected predictors.

#' Coefficients & Interpretation (Intercept) = 5.05  
#' –When all predictors are zero, the log-odds of AD are 5.05 (very high baseline probability).  
#' Log-odds = 5.05 -> Odds = e^{5.05} ≈ 156 -> Probability = 156/(1+156) ≈ 0.994

#' FunctionalAssessment = –0.48 (p < 2e-16)  
#' –Higher functional ability is strongly associated with lower AD risk.  

#' ADL = –0.44 (p < 2e-16)  
#' –Better daily-living performance corresponds to substantially reduced AD risk.  

#' MemoryComplaints1 = +2.52 (p < 2e-16)  
#' –Reporting memory complaints is a clear indicator of increased AD likelihood.  

#' BehavioralProblems1 = +2.45 (p < 2e-16)  
#' –Presence of behavioral problems strongly signals higher odds of AD.  

#' MMSE = –0.11 (p < 2e-16)  
#' –Higher cognitive test scores are linked to lower probability of AD.  

#' SleepQuality = –0.08 (p = 0.0746)  
#' –Better sleep quality suggests some protection against AD, though this is only borderline significant.    

#' CholesterolLDL = –0.003 (p = 0.159)  
#' –No clear effect after accounting for other factors.  

#' FamilyHistoryAlzheimers1 = –0.18 (p = 0.303)  
#' –No significant effect in this multivariable context.  

#' Model Fit Statistics
#' Null deviance: 1955.2 on 1504 df  
#' Residual deviance: 1097.7 on 1496 df  
#' –Large drop indicates the predictors explain a lot of variation.  
#' AIC: 1115.7  
#' –Lower AIC signals a better trade-off between fit and complexity.  

#' Convergence
#' Fisher scoring iterations: 6 
#' –The iterative fitting algorithm (IRLS) converged in 6 steps, showing stable estimates.





set.seed(65778275)
#' Define the training control for 10-fold cross-validation
train_control <- trainControl(method = "cv",  # Use cross-validation
                              number = 10,    # Specify the number of folds (k=10)
)
#' Then, run the train() function again:
model_cv <- train(
  as.factor(Diagnosis) ~ .,
  data = train_data[, c("Diagnosis", predictors)],
  method = "glm",
  family = "binomial",
  trControl = train_control
)

#' Print the cross-validation results
print(model_cv)
#' from cross-validation we can see Accuracy of 0.8544415 and Kappa of 0.6684592
#' Kappa is a metric that measures how much better classifier is compared to random guessing

#' Multicolinearity
vif(model_primary)
#' VIF values are < 5 therefore no multicolinearity is detected.

#' Convert log-odds to odds-ratios
exp(cbind(OddsRatios = coef(model_primary),   # exponentiate → odds ratios
          confint(model_primary)))            # CIs for interpretation

#' For a one-unit increase in FunctionalAssessment, the odds of the positive 
#' outcome are multiplied by approximately 0.64. This indicates that higher 
#' values of FunctionalAssessment are associated with lower odds of the 
#' positive outcome (since 0.64 < 1).





#' Checking Conditions for residuals


residuals_primary <- residuals(model_primary, type = "deviance")


#'# QQ-Plot for Residuals
# qqnorm(residuals_primary, main = "QQ-Plot of Deviance Residuals (Primary Model)")
# qqline(residuals_primary, col = "red")

car::qqPlot(residuals_primary, main = "Q-Q Plot of Residuals from Logistic Regression")

#' Interpretation of QQ-Plot:
#' This plot helps assess if the distribution of residuals follows a theoretical normal distribution.
#'  If the points fall roughly along the red line, the residuals are approximately normally distributed.
#'  Deviations from the line, especially S-shapes or curves, suggest non-normality.
#'  Points far away from the line at the ends indicate potential outliers (residuals that are much larger or smaller than expected). There are some outliers visible and even marked. Also we can observe S-shape forming and can conclude that it is not perfect.


#'# Statistical Tests on Residuals

#' Perform tests
shapiro_test_result <- shapiro.test(residuals_primary)
ttest_result <- t.test(residuals_primary)
bp_test_result <- bptest(model_primary)
tpt_result <- turning.point.test(residuals_primary)
dw_test_result <- dwtest(model_primary)


#' Table of Test Results
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

#' Fill the Interpretation column with GOOD/BAD
test_results_table$Interpretation[1] <- ifelse(shapiro_test_result$p.value > 0.05, "GOOD", "BAD")
test_results_table$Interpretation[2] <- ifelse(ttest_result$p.value > 0.05, "GOOD", "BAD")
test_results_table$Interpretation[3] <- ifelse(bp_test_result$p.value > 0.05, "GOOD", "BAD")
test_results_table$Interpretation[4] <- ifelse(tpt_result$p.value > 0.05, "GOOD", "BAD")
test_results_table$Interpretation[5] <- ifelse(is.na(dw_test_result$p.value), "N/A", ifelse(dw_test_result$p.value > 0.05, "GOOD", "BAD"))

print(test_results_table)

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
print(outcome_interpretation)




#' Predict probabilities on test set
test_probs <- predict(model_primary, newdata = test_data, type = "response")

#' ROC analysis
roc_obj_primary <- roc(test_data$Diagnosis, test_probs)
auc_primary <- auc(roc_obj_primary)
cat("AUC:", round(auc_primary, 3), "\n")

#' Area under the curve for glm model is 0.908

#' Get optimal threshold
coords_obj_primary <- coords(roc_obj_primary, "best", best.method = "youden", ret = c("threshold", "sensitivity", "specificity"))
optimal_threshold_primary <- coords_obj_primary["threshold"]
optimal_threshold_primary

#' Confusion matrix - threshold 0.5
preds_default <- factor(ifelse(test_probs > 0.5, 1, 0), levels = c(0, 1))
conf_matrix_default <- confusionMatrix(preds_default, as.factor(test_data$Diagnosis), positive = "1")
cat("\nConfusion Matrix (Threshold 0.5):\n")
print(conf_matrix_default)

optimal_threshold_primary
#' Confusion matrix - optimal threshold
preds_optimal <- factor(ifelse(test_probs > 0.3463317, 1, 0), levels = c(0, 1))
conf_matrix_optimal <- confusionMatrix(preds_optimal, as.factor(test_data$Diagnosis), positive = "1")
cat("\nConfusion Matrix (Optimal Threshold):\n")
print(conf_matrix_optimal)

#' Visualize Confusion Matrices, this funciton will be also used later with GBM
create_cm_df <- function(cm, model_name) {
  as.data.frame(cm$table) %>%
    mutate(Model = model_name)
}

cm_df_primary_default <- create_cm_df(conf_matrix_default, "GLM (Th=0.5)")
cm_df_primary_optimal <- create_cm_df(conf_matrix_optimal, "GLM (Th=0.3463317)")

cm_df_primary <- rbind(cm_df_primary_default, cm_df_primary_optimal)

ggplot(cm_df_primary, aes(x = Reference, y = Prediction, fill = Freq)) +
  geom_tile() +
  geom_text(aes(label = Freq), color = "white", size = 5) +
  scale_fill_gradient(low = "lightblue", high = "blue") +
  facet_wrap(~ Model) +
  labs(title = "Confusion Matrices - Primary Model", fill = "Count") +
  theme_minimal() +
  theme(aspect.ratio = 1)




majority_class <- names(which.max(table(data_numeric$Diagnosis)))
majority_class_count <- max(table(data_numeric$Diagnosis))
total_samples <- nrow(data_numeric)
no_information_rate <- majority_class_count / total_samples

cat("Majority Class:", majority_class, "\n")
cat("No Information Rate:", no_information_rate, "\n")

#' Logistic regression model's accuracy is higher than the No Information Rate.
#' Therefore our model is learning useful information from the predictors.




#'# Second Model - XBoost (GBM)

#' main article:
#'@article{article,
#'  author = {Dada, Emmanuel and Ibrahim Birma, Aishatu and Gora, Abdulkarim},
#'  year = {2024},
#'  month = {04},
#'  pages = {46-73},
#'  title = {Ensemble Machine Learning Algorithm for Diabetes Prediction in Maiduguri, Borno State},
#'  volume = {2},
#'  journal = {Mikailalsys Journal of Mathematics and Statistics},
#'  doi = {10.58578/mjms.v2i2.2875}
#'}

#' this study applied various ensemble methods, including GBoost, to predict diabetes .
#' in this article they used this parameters:
#' n_estimators 400
#' max_depth 5
#' Loss Squared_error
#' min_samples_split 2
#' learning_rate 0.1
#' 
#' The Gradient Boosting Regressor (GBoost) was among the models assessed, with its performance metrics reported as follows:
#' Root Mean Square Error (RMSE): 0.4833
#' Mean Square Error (MSE): 0.2336
#' Mean Absolute Error (MAE): 0.4003



#' average Accuracy for GBM cross-validation appears to be Average 94%,
#' so it performs very well. also Kappa is 0.867 which is also very good result.

train_data_gbm <- train_data
train_data_gbm$Diagnosis <- as.numeric(as.character(train_data_gbm$Diagnosis))

gbm_model <- gbm(
  Diagnosis ~ ., # Formula: Predict 'Diagnosis' using all other columns in the specified data
  data = train_data_gbm[, c("Diagnosis", predictors)], # Data frame and the columns to be used for training (Diagnosis and the selected predictors)
  distribution = "bernoulli", # logistic - Specifies the loss function to be used. "bernoulli" is appropriate for binary classification (0 or 1 outcomes like Diagnosis)
  n.trees = 400, # The number of boosting iterations to perform. Each tree is built sequentially, trying to correct the errors of the previous trees. More trees can lead to better performance but also increased risk of overfitting and longer training time.
  shrinkage = 0.1, # Also known as the learning rate. It scales the contribution of each tree. A smaller shrinkage value (e.g., 0.01, 0.1) makes the learning slower but often results in more robust models that generalize better.
  interaction.depth = 5, # The maximum depth of each tree. It controls the complexity of the interactions between predictors that the model can capture. A depth of 1 means the model is additive (no interactions), a depth of 2 allows for two-way interactions, and so on. Higher values can capture more complex relationships but also increase the risk of overfitting.
  n.minobsinnode = 10, # The minimum number of observations in the terminal nodes (leaves) of each tree. This parameter helps to regularize the model and prevent overfitting by ensuring that leaves are not too specific to small subsets of the training data.
  verbose = FALSE, # A logical value indicating whether to print progress and performance updates during the training process. FALSE means no output during training.
  cv.folds = 10
)

#' summary - variable importance
summary(gbm_model)

#' Variable Importance:
#' The summary of the GBM model indicates the relative influence of each predictor variable on the model's predictions.
#' MSE is the most important variable, with a relative influence of approximately 23.87%.
#' MMSE and ADL are also highly influential, with relative influences of around 23.85% and 20.5%.


#' Predict probabilities
pred_probs <- predict(gbm_model, newdata = test_data, n.trees = 400, type = "response")

#' classify using 0.5 threshold
pred_labels <- ifelse(pred_probs > 0.5, 1, 0)

#' Actual labels
true_labels <- test_data$Diagnosis

#' Confusion matrix
conf_mat <- confusionMatrix(as.factor(pred_labels), as.factor(true_labels), positive = "1")
print(conf_mat)

#'True Negatives (TN): (Number of actual 0s correctly predicted as 0)
#'False Negatives (FN): (Number of actual 1s incorrectly predicted as 0)
#'False Positives (FP): (Number of actual 0s incorrectly predicted as 1)
#'True Positives (TP): (Number of actual 1s correctly predicted as 1)
#'Accuracy: (93% of predictions were correct)
#'95% Confidence Interval (CI) for Accuracy: (0.9164, 0.9553)
#'No Information Rate (NIR): (Accuracy if the model always predicted the majority class)
#'P-Value [Acc > NIR]: < 2e-16 (Highly significant, indicating the model's accuracy is significantly better than random guessing)
#'Kappa: 0.867 (Measure of agreement between predicted and actual classifications, with 1 being perfect agreement)
#'Mcnemar's Test P-Value: 0.02064 (P-value for McNemar's chi-squared test, testing for symmetry of errors. A value > 0.05 suggests no significant difference in the types of errors)
#'Sensitivity (Recall, True Positive Rate): (Proportion of actual 1s correctly identified)
#'Specificity (True Negative Rate): (Proportion of actual 0s correctly identified)
#'Positive Predictive Value (Precision): (Proportion of predicted 1s that were actually 1s)
#'Negative Predictive Value: (Proportion of predicted 0s that were actually 0s)
#'Prevalence: (Proportion of the positive class (1) in the test set)
#'Detection Rate: (Proportion of the positive class correctly identified out of the total number of instances)
#'Detection Prevalence: (Proportion of instances predicted as positive)
#'Balanced Accuracy: (Average of sensitivity and specificity)
#'Positive' Class: 1 (Indicates that class '1' was treated as the positive class for sensitivity and specificity calculations)


cm_df <- as.data.frame(conf_mat$table)

#' Rename columns for clarity
cm_df <- cm_df %>%
  rename(Predicted = Prediction, Actual = Reference, Count = Freq)

#' Create the plot
ggplot(cm_df, aes(x = Actual, y = Predicted, fill = Count)) +
  geom_tile() +
  geom_text(aes(label = Count), color = "white", size = 5) +
  scale_fill_gradient(low = "lightblue", high = "blue") +
  labs(title = "Confusion Matrix", fill = "Count") +
  theme_minimal() +
  theme(aspect.ratio = 1)


#' AUC
roc_obj_gboost <- roc(true_labels, pred_probs)
auc_gboost <- auc(roc_obj_gboost)
cat("AUC:", auc_gboost, "\n")

#' Area Under the ROC Curve (AUC) - 95.6%:



coords_obj_gboost <- coords(roc_obj_gboost, "best", best.method = "youden", ret = c("threshold", "sensitivity", "specificity"))
optimal_threshold_gboost <- coords_obj_gboost["threshold"]
optimal_threshold_gboost


preds_optimal <- factor(ifelse(pred_probs > 0.2428418, 1, 0), levels = c(0, 1))
conf_matrix_optimal <- confusionMatrix(preds_optimal, as.factor(test_data$Diagnosis), positive = "1")
cat("\nConfusion Matrix (Optimal Threshold):\n")
print(conf_matrix_optimal)



cm_df_default <- create_cm_df(conf_mat, "Primary (Th=0.5)")
cm_df_optimal <- create_cm_df(conf_matrix_optimal, "Primary (Th=0.2641755)")

cm_df_primary <- rbind(cm_df_default, cm_df_optimal)

ggplot(cm_df_primary, aes(x = Reference, y = Prediction, fill = Freq)) +
  geom_tile() +
  geom_text(aes(label = Freq), color = "white", size = 5) +
  scale_fill_gradient(low = "lightblue", high = "blue") +
  facet_wrap(~ Model) +
  labs(title = "Confusion Matrices - Primary Model", fill = "Count") +
  theme_minimal() +
  theme(aspect.ratio = 1)



#'# Comparison of models



#' Visualize ROC Curves
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
  annotate("text", x = 0.75, y = 0.25, label = paste("AUC (GLM) =", round(auc_primary, 3)), color = "cyan4") +
  annotate("text", x = 0.75, y = 0.20, label = paste("AUC (GBoost) =", round(auc_gboost, 3)), color = "red")
