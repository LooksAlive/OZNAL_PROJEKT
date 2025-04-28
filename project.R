getwd()

library(tidyverse)
library(tidymodels)
#library(ggplot2)
library(magrittr)
library(car)
library(caret)
library(gridExtra)
library(GGally)
library(pROC)
library(corrplot)
library(ggmosaic)
library(dplyr)
library(ggcorrplot)
library(MASS)
library(randomForest)
library(infotheo)  # MI
library(entropy)   # MI
library(rpart)
library(rpart.plot)
library(patchwork) # combining plots
library(lmtest) # Breusch-Pagan test for homoscedasticity
library(car)    # Levene's test for homoscedasticit
library(latexpdf)



data <- read.csv("alzheimers_disease_data.csv")

#'# 1. Initial Data Exploration

#' Drop the 'DoctorInCharge' column since it is empty and no use to us.
data <- data %>% dplyr::select(-DoctorInCharge, -PatientID)

str(data)
summary(data)

sum(is.na(data))
colSums(is.na(data))

table(data$Diagnosis)
prop.table(table(data$Diagnosis))
#' disease distribution is biased - more negatives

#' convert to factors
for (col_name in names(data)) {
  if (is.numeric(data[[col_name]]) && length(unique(data[[col_name]])) <= 5) {
    data <- data %>%
      mutate(!!sym(col_name) := as.factor(!!sym(col_name)))
  }
}

str(data)

data %>%
  select_if(is.numeric) %>%
  pivot_longer(everything(), names_to = "variable", values_to = "value") %>%
  ggplot(aes(x = value)) +
  geom_histogram(fill = "steelblue", color = "black") +
  facet_wrap(~ variable, scales = "free_x") +
  labs(title = "Histograms of Numerical Attributes", x = "Value", y = "Frequency") +
  theme_minimal()

data %>%
  select_if(is.factor) %>%
  pivot_longer(everything(), names_to = "variable", values_to = "value") %>%
  ggplot(aes(x = value, fill = value)) +
  geom_bar() +
  facet_wrap(~ variable, scales = "free_x") +
  labs(title = "Bar Plots of Factor Attributes", x = "Category", y = "Count") +
  theme_minimal() +
  theme(axis.text.x = element_text(angle = 45, hjust = 1))

#' This was an interesting move to show distributions like this, we can observe
#' for example that positive diagnosis resonates in some attributes:













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

numeric_vars <- data %>% 
  select_if(is.numeric)
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



#' Normality
normality_p_values <- apply(numeric_vars, 2, function(x) {
    shapiro.test(x)$p.value
})
cat("- Normality of predictors: Checking with Shapiro-Wilk test (for applicable numeric predictors):\n")
print(sort(normality_p_values))
cat("  (P-values < 0.05 suggest a deviation from normality. Consider transformations if needed.)\n")

#' Not normally distributed



numeric_preds <- numeric_vars
response_var <- as.numeric(data$Diagnosis) - 1 # Convert to 0/1 for modeling

cat("--- Homoscedasticity Tests for Each Numeric Predictor ---\n\n")

homoscedasticity_results <- list()

for (col in names(numeric_preds)) {
  cat(paste("--- Testing homoscedasticity for predictor:", col, "---\n"))
  
  formula_str <- paste("response_var ~", col)
  model_bp <- lm(formula_str, data = data.frame(response_var = response_var, numeric_preds))
  
  # Breusch-Pagan Test
  bp_test <- bptest(model_bp)
  bp_p_value <- bp_test$p.value
  cat(paste("Breusch-Pagan Test P-value:", round(bp_p_value, 4), "\n"))
  if (bp_p_value < 0.05) {
    cat("Interpretation: The Breusch-Pagan test suggests evidence of heteroscedasticity (non-constant variance of errors) for this predictor.\n")
  } else {
    cat("Interpretation: The Breusch-Pagan test does not provide strong evidence against homoscedasticity (constant variance of errors) for this predictor.\n")
  }
  cat("\n")
  
  # Levene's Test
  levene_df <- data.frame(y = data[[col]], group = data$Diagnosis)
  levene_test <- leveneTest(y ~ group, data = levene_df)
  levene_p_value <- levene_test$`Pr(>F)`[1]
  cat(paste("Levene's Test P-value:", round(levene_p_value, 4), "\n"))
  if (levene_p_value < 0.05) {
    cat("Interpretation: Levene's test suggests that the variances of the predictor variable are significantly different between the two Diagnosis groups.\n")
  } else {
    cat("Interpretation: Levene's test does not provide strong evidence that the variances of the predictor variable are different between the two Diagnosis groups.\n")
  }
  cat("\n")
  
  homoscedasticity_results[[col]] <- list(Breusch_Pagan_P_value = bp_p_value, Levene_P_value = levene_p_value)
}

print(homoscedasticity_results)



cor_matrix <- cor(data %>% select_if(is.numeric))
high_corr <- findCorrelation(cor_matrix, cutoff = 0.75) 
if (length(high_corr) > 0) {
  high_corr_names <- colnames(cor_matrix)[high_corr]
  cat(paste("- Potential multicollinearity in:", paste(high_corr_names, collapse = ", "), "\n"))
} else {
  cat("- No severe multicollinearity detected (based on correlation).\n")
}






#'# Summary Statistics for Numerical Attributes

data %>%
  select_if(is.numeric) %>%
  summary()

# --- Relationship between Two Categorical Attributes (e.g., Gender and Diagnosis) ---
if ("Gender" %in% names(data) && is.factor(data$Gender) &&
    "Diagnosis" %in% names(data) && is.factor(data$Diagnosis)) {
  print("\n--- Cross-Tabulation: Gender vs. Diagnosis ---")
  cross_tab_gender_diag <- table(data$Gender, data$Diagnosis)
  print(cross_tab_gender_diag)
  print("\nChi-squared Test: Gender vs. Diagnosis")
  print(chisq.test(cross_tab_gender_diag))
}












categorical_vars <- c("Gender", "Ethnicity", "EducationLevel", "Smoking", 
                      "FamilyHistoryAlzheimers", "CardiovascularDisease", 
                      "Diabetes", "Depression", "Diagnosis")

health_metrics <- c("BMI", "SystolicBP", "DiastolicBP", "CholesterolTotal", 
                    "CholesterolLDL", "CholesterolHDL", "MMSE", "FunctionalAssessment", 
                    "SleepQuality")

data_long <- data %>%
  dplyr::select(Diagnosis, all_of(health_metrics)) %>%
  pivot_longer(cols = all_of(health_metrics), 
               names_to = "Metric", 
               values_to = "Value")

ggplot(data_long, aes(x = Metric, y = Value, fill = factor(Diagnosis))) +
  geom_boxplot() +
  facet_wrap(~ Metric, scales = "free") +
  labs(title = "Health Metrics by Diagnosis Status",
       x = NULL,
       y = "Value",
       fill = "Diagnosis") +
  theme_minimal() +
  theme(axis.text.x = element_text(angle = 45, hjust = 1))

#' Perform t-tests to compare means between diagnosis groups - 
#' draw conslusion from statistics now instead of visual boxplots 
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
















#'# 2. Logistic Regression with all variables
#' Convert factors to numeric for modeling
data_model <- data
for (var in cat_vars) {
  data_model[[var]] <- as.numeric(as.character(data_model[[var]]))
}
data_model$Diagnosis <- as.factor(data_model$Diagnosis)

full_model <- glm(Diagnosis ~ ., data = data_model, family = binomial)
summary_model <- summary(full_model)

coef_table <- as.data.frame(summary_model$coefficients)
coef_table$Variable <- rownames(coef_table)
rownames(coef_table) <- NULL
coef_table <- coef_table[order(abs(coef_table$`z value`), decreasing = TRUE), ]
coef_table <- coef_table[2:nrow(coef_table), ] # Remove intercept
coef_table$p_value <- coef_table$`Pr(>|z|)`
coef_table$significance <- ifelse(coef_table$p_value < 0.05, "*", "")
coef_table$significance[coef_table$p_value < 0.01] <- "**"
coef_table$significance[coef_table$p_value < 0.001] <- "***"

print("Top predictors based on logistic regression:")
head(coef_table[, c("Variable", "Estimate", "z value", "p_value", "significance")], 15)

#' Stepwise Selection to identify key predictors
step_model <- stepAIC(full_model, direction = "both", trace = FALSE)
final_formula <- formula(step_model)
print("Variables selected by stepwise selection:")
print(final_formula)

#' Random Forest for feature importance
set.seed(123)
rf_model <- randomForest(Diagnosis ~ ., data = data_model, importance = TRUE)
rf_model
importance_df <- as.data.frame(importance(rf_model))
importance_df$Variable <- rownames(importance_df)
rownames(importance_df) <- NULL
importance_df <- importance_df[order(importance_df$MeanDecreaseGini, decreasing = TRUE), ]

print("Top predictors based on Random Forest importance:")
head(importance_df[, c("Variable", "MeanDecreaseAccuracy", "MeanDecreaseGini")], 15)

#' Visualize feature importance
ggplot(head(importance_df, 15), aes(x = reorder(Variable, MeanDecreaseGini), y = MeanDecreaseGini)) +
  geom_bar(stat = "identity", fill = "steelblue") +
  coord_flip() +
  theme_minimal() +
  labs(title = "Top 15 Predictors by Random Forest", 
       x = "Variables", 
       y = "Mean Decrease Gini")








#'# Information Value analysis
library(Information)
iv_data <- data
iv_data$Diagnosis <- as.numeric(as.character(iv_data$Diagnosis))

iv_results <- data.frame(
  Variable = character(),
  IV = numeric(),
  stringsAsFactors = FALSE
)

for (var in names(iv_data)[names(iv_data) != "Diagnosis"]) {
  tryCatch({
    iv_obj <- create_infotables(data = iv_data, y = "Diagnosis", bins = 10, parallel = FALSE)
    var_iv <- iv_obj$Tables[[var]]$IV
    iv_results <- rbind(iv_results, data.frame(Variable = var, IV = var_iv, stringsAsFactors = FALSE))
  }, error = function(e) {
    # Skip variables that cause errors
  })
}

iv_results <- iv_results[order(iv_results$IV, decreasing = TRUE), ]
print("Predictors ranked by Information Value:")
print(iv_results)








#' Cross-validated performance metrics for key predictors
set.seed(123)
cv_formula <- final_formula

evaluate_model <- function(formula, data) {
  set.seed(123)
  # Create 10-fold CV
  folds <- createFolds(data$Diagnosis, k = 10)
  cv_results <- data.frame(Accuracy = numeric(), Sensitivity = numeric(), 
                           Specificity = numeric(), AUC = numeric())
  
  for (i in 1:length(folds)) {
    # Split data
    test_indices <- folds[[i]]
    train_data <- data[-test_indices, ]
    test_data <- data[test_indices, ]
    
    # Train model
    model <- glm(formula, data = train_data, family = binomial)
    
    # Predict
    predictions <- predict(model, test_data, type = "response")
    pred_class <- ifelse(predictions > 0.5, 1, 0)
    
    # Evaluate
    conf_matrix <- confusionMatrix(factor(pred_class), test_data$Diagnosis, positive = "1")
    roc_obj <- roc(as.numeric(test_data$Diagnosis) - 1, predictions)
    
    # Store results
    cv_results <- rbind(cv_results, data.frame(
      Accuracy = conf_matrix$overall["Accuracy"],
      Sensitivity = conf_matrix$byClass["Sensitivity"],
      Specificity = conf_matrix$byClass["Specificity"],
      AUC = auc(roc_obj)
    ))
  }
  
  # Return average metrics
  return(colMeans(cv_results))
}

cv_metrics <- evaluate_model(cv_formula, data_model)
print("Cross-validated performance metrics using selected features:")
print(cv_metrics)








#'# Final logistic regression model


top_predictors <- c("MMSE", "BehavioralProblems", "FunctionalAssessment", 
                    "MemoryComplaints", "CholesterolHDL",
                    "SleepQuality", "ADL")

model_data <- data %>% 
  dplyr::select(all_of(c(top_predictors, "Diagnosis")))

model_data <- na.omit(model_data)

set.seed(123)
train_index <- createDataPartition(model_data$Diagnosis, p = 0.7, list = FALSE)
train_data <- model_data[train_index, ]
test_data <- model_data[-train_index, ]

log_model <- glm(Diagnosis ~ ., data = train_data, family = binomial)

summary_log <- summary(log_model)
print(summary_log)

test_probs <- predict(log_model, newdata = test_data, type = "response")

roc_obj <- roc(test_data$Diagnosis, test_probs)
print(paste("AUC:", round(auc(roc_obj), 3)))

coords_obj <- coords(roc_obj, "best", best.method = "youden")
optimal_threshold <- coords_obj$threshold
print(paste("Optimal threshold:", round(optimal_threshold, 3)))

test_preds <- ifelse(test_probs > optimal_threshold, 1, 0)
test_preds <- factor(test_preds, levels = levels(test_data$Diagnosis))

conf_matrix <- confusionMatrix(test_preds, test_data$Diagnosis, positive = "1")
print(conf_matrix)

plot(roc_obj, main = "ROC Curve for Logistic Regression Model")
points(coords_obj$specificity, coords_obj$sensitivity, pch = 19, col = "red")
text(coords_obj$specificity, coords_obj$sensitivity, 
     labels = paste("Threshold =", round(optimal_threshold, 2)), pos = 4)

pred_df <- data.frame(
  Actual = test_data$Diagnosis,
  Probability = test_probs
)

ggplot(pred_df, aes(x = Probability, fill = Actual)) +
  geom_density(alpha = 0.5) +
  geom_vline(xintercept = optimal_threshold, linetype = "dashed", color = "red") +
  annotate("text", x = optimal_threshold + 0.05, y = 0.5, 
           label = paste("Threshold =", round(optimal_threshold, 2))) +
  labs(title = "Distribution of Predicted Probabilities by Actual Class",
       x = "Predicted Probability", 
       y = "Density") +
  theme_minimal()



ggplot(log_model$data, aes(sample = residuals(log_model, type = "deviance"))) +
  stat_qq() +
  stat_qq_line() +
  labs(title = "QQ Plot of Deviance Residuals",
       x = "Theoretical Quantiles",
       y = "Sample Quantiles") +
  theme_minimal()


ggplot(log_model$data, aes(x = fitted(log_model), y = residuals(log_model, type = "deviance"))) +
  geom_point(alpha = 0.6) +
  geom_hline(yintercept = 0, linetype = "dashed") +
  labs(title = "Deviance Residuals vs. Fitted Values",
       x = "Fitted Probabilities",
       y = "Deviance Residuals") +
  theme_minimal()


#if (ncol(data) > 3) {
#  ggpairs(data %>% dplyr::select(Age, Gender, Ethnicity, MMSE, FunctionalAssessment, Diagnosis),
#          aes(color = factor(Diagnosis)))
#}