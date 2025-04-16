# Set working directory (already done in your code)
getwd()

# Load necessary libraries (already done in your code)
library(tidyverse)
library(tidymodels)
library(ggplot2)
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
library(infotheo)  # For mutual information calculations
library(entropy)   # Alternative approach for MI
library(rpart)
library(rpart.plot)

# Load the dataset (already done in your code)
data <- read.csv("alzheimers_disease_data.csv")

# Drop the 'DoctorInCharge' column
data <- data %>% dplyr::select(-DoctorInCharge, -PatientID)
# --- 1. Initial Data Exploration ---

# Print the first few rows of the data
#head(data)

# Get a summary of the data structure and data types
str(data)

# Get descriptive statistics for numerical variables
summary(data)

# Check for missing values
sum(is.na(data))
colSums(is.na(data))

# We do not have any missing data

# Check the distribution of the target variable 'Diagnosis'
table(data$Diagnosis)
# disease distribution is biased - more negatives
prop.table(table(data$Diagnosis))

for (col_name in names(data)) {
  if (is.numeric(data[[col_name]]) && length(unique(data[[col_name]])) <= 5) {
    data <- data %>%
      mutate(!!sym(col_name) := as.factor(!!sym(col_name)))
  }
}

str(data)

# --- Histograms for Numerical Variables ---
data %>%
  select_if(is.numeric) %>%
  pivot_longer(everything(), names_to = "variable", values_to = "value") %>%
  ggplot(aes(x = value)) +
  geom_histogram(fill = "steelblue", color = "black") +
  facet_wrap(~ variable, scales = "free_x") +
  labs(title = "Histograms of Numerical Attributes", x = "Value", y = "Frequency") +
  theme_minimal()

# --- Bar Plots for Factor Variables ---
data %>%
  select_if(is.factor) %>%
  pivot_longer(everything(), names_to = "variable", values_to = "value") %>%
  ggplot(aes(x = value, fill = value)) +
  geom_bar() +
  facet_wrap(~ variable, scales = "free_x") +
  labs(title = "Bar Plots of Factor Attributes", x = "Category", y = "Count") +
  theme_minimal() +
  theme(axis.text.x = element_text(angle = 45, hjust = 1))




















# --- Gender vs. Diagnosis ---
ggplot(data, aes(x = Gender, fill = Diagnosis)) +
  geom_bar(position = "dodge") +
  labs(title = "Gender vs. Diagnosis", x = "Gender", y = "Count") +
  theme_minimal() +
  scale_x_discrete(labels = c("Male", "Female")) # Assuming 0=Male, 1=Female

# --- Age vs. Diagnosis ---
ggplot(data, aes(x = Diagnosis, y = Age, fill = Diagnosis)) +
  geom_boxplot() +
  labs(title = "Age vs. Diagnosis", x = "Diagnosis", y = "Age") +
  theme_minimal() +
  scale_x_discrete(labels = c("No Alzheimer's", "Alzheimer's")) # Assuming 0=No, 1=Yes

# --- Gender & Age vs. Diagnosis ---
#ggplot(data, aes(x = Age, y = Gender, color = Diagnosis)) +
#  geom_point() +
#  labs(title = "Gender & Age vs. Diagnosis", x = "Age", y = "Gender") +
#  theme_minimal() +
#  scale_y_continuous(breaks = c(0, 1), labels = c("Male", "Female")) +
#  scale_color_discrete(labels = c("No Alzheimer's", "Alzheimer's"))

# --- Smoking vs. Diagnosis ---
ggplot(data, aes(x = Smoking, fill = Diagnosis)) +
  geom_bar(position = "dodge") +
  labs(title = "Smoking vs. Diagnosis", x = "Smoking", y = "Count") +
  theme_minimal() +
  scale_x_discrete(labels = c("No", "Yes")) # Assuming 0=No, 1=Yes

# --- PhysicalActivity vs. Diagnosis ---
ggplot(data, aes(x = Diagnosis, y = PhysicalActivity, fill = Diagnosis)) +
  geom_boxplot() +
  labs(title = "Physical Activity vs. Diagnosis", x = "Diagnosis", y = "Physical Activity (hours/week)") +
  theme_minimal() +
  scale_x_discrete(labels = c("No Alzheimer's", "Alzheimer's"))

# --- DietQuality vs. Diagnosis ---
ggplot(data, aes(x = Diagnosis, y = DietQuality, fill = Diagnosis)) +
  geom_boxplot() +
  labs(title = "Diet Quality vs. Diagnosis", x = "Diagnosis", y = "Diet Quality (score 0-10)") +
  theme_minimal() +
  scale_x_discrete(labels = c("No Alzheimer's", "Alzheimer's"))

# --- SleepQuality vs. Diagnosis ---
ggplot(data, aes(x = Diagnosis, y = SleepQuality, fill = Diagnosis)) +
  geom_boxplot() +
  labs(title = "Sleep Quality vs. Diagnosis", x = "Diagnosis", y = "Sleep Quality (score 4-10)") +
  theme_minimal() +
  scale_x_discrete(labels = c("No Alzheimer's", "Alzheimer's"))

# --- Other Simple Visualizations ---

# --- BMI Distribution by Diagnosis ---
ggplot(data, aes(x = Diagnosis, y = BMI, fill = Diagnosis)) +
  geom_boxplot() +
  labs(title = "BMI vs. Diagnosis", x = "Diagnosis", y = "BMI") +
  theme_minimal() +
  scale_x_discrete(labels = c("No Alzheimer's", "Alzheimer's"))

# --- MMSE (Cognitive Assessment) by Diagnosis ---
ggplot(data, aes(x = Diagnosis, y = MMSE, fill = Diagnosis)) +
  geom_boxplot() +
  labs(title = "MMSE vs. Diagnosis", x = "Diagnosis", y = "MMSE Score") +
  theme_minimal() +
  scale_x_discrete(labels = c("No Alzheimer's", "Alzheimer's"))

# --- Correlation Heatmap (for numerical variables) ---
numerical_data <- data %>% select_if(is.numeric)
if (ncol(numerical_data) > 1) {
  correlation_matrix <- cor(numerical_data)
  correlation_df <- as.data.frame(correlation_matrix) %>%
    mutate(Var1 = rownames(.)) %>%
    pivot_longer(cols = -Var1, names_to = "Var2", values_to = "Correlation")
  
  ggplot(correlation_df, aes(x = Var1, y = Var2, fill = Correlation)) +
    geom_tile() +
    scale_fill_gradient2(low = "blue", mid = "white", high = "red", limits = c(-1, 1)) +
    labs(title = "Correlation Heatmap of Numerical Attributes") +
    theme_minimal() +
    theme(axis.text.x = element_text(angle = 45, hjust = 1),
          axis.title.x = element_blank(),
          axis.title.y = element_blank())
} else {
  print("Not enough numerical variables for a correlation heatmap.")
}







# --- Summary Statistics for Numerical Attributes ---
data %>%
  select_if(is.numeric) %>%
  summary()

# --- Frequency Tables for Factor Attributes ---
data %>%
  select_if(is.factor) %>%
  map(~ table(.) %>% prop.table() %>% round(3) %>% as.data.frame() %>% print())

# --- Relationship between Two Categorical Attributes (e.g., Gender and Diagnosis) ---
if ("Gender" %in% names(data) && is.factor(data$Gender) &&
    "Diagnosis" %in% names(data) && is.factor(data$Diagnosis)) {
  print("\n--- Cross-Tabulation: Gender vs. Diagnosis ---")
  cross_tab_gender_diag <- table(data$Gender, data$Diagnosis)
  print(cross_tab_gender_diag)
  print("\nChi-squared Test: Gender vs. Diagnosis")
  print(chisq.test(cross_tab_gender_diag))
}

# --- Relationship between a Numerical and a Categorical Attribute (e.g., Age and Diagnosis) ---
if ("Age" %in% names(data) && is.numeric(data$Age) &&
    "Diagnosis" %in% names(data) && is.factor(data$Diagnosis)) {
  print("\n--- Summary Statistics of Age by Diagnosis ---")
  print(data %>% group_by(Diagnosis) %>% summarize(
    Mean_Age = mean(Age, na.rm = TRUE),
    Median_Age = median(Age, na.rm = TRUE),
    SD_Age = sd(Age, na.rm = TRUE),
    Min_Age = min(Age, na.rm = TRUE),
    Max_Age = max(Age, na.rm = TRUE),
    n = n()
  ))
  
  # --- T-test (assuming two levels in Diagnosis) ---
  if (nlevels(data$Diagnosis) == 2) {
    print("\n--- T-test: Age vs. Diagnosis ---")
    print(t.test(Age ~ Diagnosis, data = data))
  } else {
    print("\nNote: Diagnosis has more than 2 levels, skipping t-test.")
  }
  
  # --- ANOVA (for more than two levels in Diagnosis) ---
  if (nlevels(data$Diagnosis) > 2) {
    print("\n--- ANOVA: Age vs. Diagnosis ---")
    anova_model <- aov(Age ~ Diagnosis, data = data)
    print(summary(anova_model))
  }
}
















# Extract numerical variables for correlation analysis
numeric_vars <- data %>% 
  select_if(is.numeric)

# Calculate correlation matrix
cor_matrix <- cor(numeric_vars, use = "complete.obs")

# Visualize correlation matrix
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

# Alternative correlation visualization with ggcorrplot
ggcorrplot(cor_matrix, 
           hc.order = TRUE, 
           type = "lower",
           lab = TRUE, 
           lab_size = 2,
           title = "Correlation Heatmap") +
  theme(axis.text.x = element_text(angle = 45, hjust = 1, size = 8),
        axis.text.y = element_text(size = 8))


# Boxplots for MMSE by categorical variables
categorical_vars <- c("Gender", "Ethnicity", "EducationLevel", "Smoking", 
                      "FamilyHistoryAlzheimers", "CardiovascularDisease", 
                      "Diabetes", "Depression", "Diagnosis")

# Compare health metrics between diagnosis groups
# Create a paired boxplot
health_metrics <- c("BMI", "SystolicBP", "DiastolicBP", "CholesterolTotal", 
                    "CholesterolLDL", "CholesterolHDL", "MMSE", "FunctionalAssessment")

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

# Perform t-tests to compare means between diagnosis groups
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

# Create a correlation network plot for key variables
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
















# 1. Univariate Analysis: Chi-square for categorical variables & t-tests for numeric variables
# Create empty dataframes to store results
cat_results <- data.frame(
  Variable = character(),
  ChiSq = numeric(),
  p_value = numeric(),
  stringsAsFactors = FALSE
)

num_results <- data.frame(
  Variable = character(),
  t_value = numeric(),
  p_value = numeric(),
  stringsAsFactors = FALSE
)

# Identify categorical and numerical variables
cat_vars <- names(data)[sapply(data, is.factor)]
cat_vars <- cat_vars[cat_vars != "Diagnosis"] # Remove the target variable
num_vars <- names(data)[sapply(data, is.numeric)]

# Chi-square tests for categorical predictors
for (var in cat_vars) {
  test <- chisq.test(data[[var]], data$Diagnosis)
  cat_results <- rbind(cat_results, data.frame(
    Variable = var,
    ChiSq = round(test$statistic, 3),
    p_value = round(test$p.value, 5),
    stringsAsFactors = FALSE
  ))
}

# Sort by p-value
cat_results <- cat_results[order(cat_results$p_value), ]
cat_results$significance <- ifelse(cat_results$p_value < 0.05, "*", "")
cat_results$significance[cat_results$p_value < 0.01] <- "**"
cat_results$significance[cat_results$p_value < 0.001] <- "***"

# T-tests for numerical predictors
for (var in num_vars) {
  formula_str <- paste(var, "~ Diagnosis")
  test <- t.test(as.formula(formula_str), data = data)
  num_results <- rbind(num_results, data.frame(
    Variable = var,
    t_value = round(test$statistic, 3),
    p_value = round(test$p.value, 5),
    stringsAsFactors = FALSE
  ))
}

# Sort by p-value
num_results <- num_results[order(num_results$p_value), ]
num_results$significance <- ifelse(num_results$p_value < 0.05, "*", "")
num_results$significance[num_results$p_value < 0.01] <- "**"
num_results$significance[num_results$p_value < 0.001] <- "***"

# Print the top significant predictors
print("Top categorical predictors based on Chi-square tests:")
print(cat_results)

print("Top numerical predictors based on t-tests:")
print(num_results)

# 2. Logistic Regression with all variables
# Convert factors to numeric for modeling
data_model <- data
for (var in cat_vars) {
  data_model[[var]] <- as.numeric(as.character(data_model[[var]]))
}
data_model$Diagnosis <- as.factor(data_model$Diagnosis)

# Build the logistic regression model
full_model <- glm(Diagnosis ~ ., data = data_model, family = binomial)
summary_model <- summary(full_model)

# Extract important coefficients
coef_table <- as.data.frame(summary_model$coefficients)
coef_table$Variable <- rownames(coef_table)
rownames(coef_table) <- NULL
coef_table <- coef_table[order(abs(coef_table$`z value`), decreasing = TRUE), ]
coef_table <- coef_table[2:nrow(coef_table), ] # Remove intercept
coef_table$p_value <- coef_table$`Pr(>|z|)`
coef_table$significance <- ifelse(coef_table$p_value < 0.05, "*", "")
coef_table$significance[coef_table$p_value < 0.01] <- "**"
coef_table$significance[coef_table$p_value < 0.001] <- "***"

# Print top coefficients
print("Top predictors based on logistic regression:")
head(coef_table[, c("Variable", "Estimate", "z value", "p_value", "significance")], 15)

# 3. Stepwise Selection to identify key predictors
step_model <- stepAIC(full_model, direction = "both", trace = FALSE)
final_formula <- formula(step_model)
print("Variables selected by stepwise selection:")
print(final_formula)

# 4. Random Forest for feature importance
set.seed(123)
rf_model <- randomForest(Diagnosis ~ ., data = data_model, importance = TRUE)
rf_model
importance_df <- as.data.frame(importance(rf_model))
importance_df$Variable <- rownames(importance_df)
rownames(importance_df) <- NULL
importance_df <- importance_df[order(importance_df$MeanDecreaseGini, decreasing = TRUE), ]

print("Top predictors based on Random Forest importance:")
head(importance_df[, c("Variable", "MeanDecreaseAccuracy", "MeanDecreaseGini")], 15)

# 5. Visualize feature importance
# Plot Mean Decrease Gini
ggplot(head(importance_df, 15), aes(x = reorder(Variable, MeanDecreaseGini), y = MeanDecreaseGini)) +
  geom_bar(stat = "identity", fill = "steelblue") +
  coord_flip() +
  theme_minimal() +
  labs(title = "Top 15 Predictors by Random Forest", 
       x = "Variables", 
       y = "Mean Decrease Gini")

# Plot logistic regression coefficients (standardized)
# Create standardized data for coefficient comparison
data_std <- data_model
num_vars <- names(data_std)[sapply(data_std, is.numeric)]
for (var in num_vars) {
  data_std[[var]] <- scale(data_std[[var]])
}

# Build standardized model
std_model <- glm(Diagnosis ~ ., data = data_std, family = binomial)
std_coef_table <- as.data.frame(summary(std_model)$coefficients)
std_coef_table$Variable <- rownames(std_coef_table)
rownames(std_coef_table) <- NULL
std_coef_table <- std_coef_table[2:nrow(std_coef_table), ] # Remove intercept
std_coef_table <- std_coef_table[order(abs(std_coef_table$Estimate), decreasing = TRUE), ]

# Plot top standardized coefficients
ggplot(head(std_coef_table, 15), aes(x = reorder(Variable, abs(Estimate)), y = Estimate)) +
  geom_bar(stat = "identity", fill = "darkgreen") +
  coord_flip() +
  theme_minimal() +
  labs(title = "Top 15 Predictors by Logistic Regression", 
       x = "Variables", 
       y = "Standardized Coefficient")

# 6. Information Value analysis
# Calculate information value for categorical variables
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

# 7. ROC curves for individual predictors
# Function to calculate AUC for each predictor
calculate_auc <- function(var, data) {
  if(is.factor(data[[var]])) {
    # For categorical variables, use as.numeric
    predictor <- as.numeric(data[[var]])
  } else {
    predictor <- data[[var]]
  }
  
  tryCatch({
    roc_obj <- roc(data$Diagnosis, predictor)
    return(auc(roc_obj))
  }, error = function(e) {
    return(NA)
  })
}

# Calculate AUC for each predictor
all_vars <- names(data)[names(data) != "Diagnosis"]
auc_results <- data.frame(
  Variable = all_vars,
  AUC = sapply(all_vars, calculate_auc, data = data),
  stringsAsFactors = FALSE
)

auc_results <- auc_results[order(abs(auc_results$AUC - 0.5), decreasing = TRUE), ]
auc_results$AUC <- round(auc_results$AUC, 3)

print("Predictors ranked by AUC (distance from 0.5):")
print(head(auc_results, 15))

# 8. Plot ROC curves for top predictors
top_predictors <- head(auc_results$Variable, 5)
colors <- c("red", "blue", "green", "purple", "orange")

plot(x = c(0, 1), y = c(0, 1), type = "l", lty = 2, 
     xlab = "False Positive Rate", ylab = "True Positive Rate",
     main = "ROC Curves for Top Predictors")

for (i in 1:length(top_predictors)) {
  var <- top_predictors[i]
  if(is.factor(data[[var]])) {
    predictor <- as.numeric(data[[var]])
  } else {
    predictor <- data[[var]]
  }
  
  tryCatch({
    roc_obj <- roc(data$Diagnosis, predictor)
    plot(roc_obj, add = TRUE, col = colors[i])
  }, error = function(e) {
    # Skip if error
  })
}

legend("bottomright", legend = top_predictors, col = colors[1:length(top_predictors)], lwd = 2)

# 9. Cross-validated performance metrics for key predictors
set.seed(123)
# Create a formula with the top features from stepwise selection
cv_formula <- final_formula

# Function to evaluate model performance
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

# Evaluate the model with selected features
cv_metrics <- evaluate_model(cv_formula, data_model)
print("Cross-validated performance metrics using selected features:")
print(cv_metrics)

# 10. Calculate Variable Inflation Factors to check for multicollinearity
# Among the selected variables
selected_vars <- all.vars(cv_formula)[-1]  # Remove response variable
selected_formula <- as.formula(paste("Diagnosis ~", paste(selected_vars, collapse = " + ")))

# Create a linear model for VIF calculation
lm_model <- lm(selected_formula, data = data_model)
vif_values <- vif(lm_model)
vif_df <- data.frame(
  Variable = names(vif_values),
  VIF = vif_values,
  stringsAsFactors = FALSE
)
vif_df <- vif_df[order(vif_df$VIF, decreasing = TRUE), ]

print("Variance Inflation Factors for selected predictors:")
print(vif_df)
print("Variables with VIF > 5 may indicate multicollinearity issues")

# Summary of key findings
print("=== SUMMARY OF KEY PREDICTORS FOR DIAGNOSIS ===")
print("Top 5 predictors by Random Forest importance:")
print(head(importance_df$Variable, 5))

print("Top 5 predictors by Logistic Regression significance:")
print(head(coef_table$Variable, 5))

print("Variables selected by stepwise selection:")
print(selected_vars)

print("Top 5 predictors by univariate analysis:")
cat_top <- head(cat_results$Variable, 3)
num_top <- head(num_results$Variable, 2)
print(c(cat_top, num_top))













































# Corrected ROC curve analysis
# Function to properly calculate AUC for each predictor
calculate_correct_auc <- function(var, data) {
  if(is.factor(data[[var]])) {
    # For categorical variables, convert appropriately
    predictor <- as.numeric(as.character(data[[var]]))
  } else {
    predictor <- data[[var]]
  }
  
  # Get the target as binary (0/1)
  target <- as.numeric(as.character(data$Diagnosis))
  
  tryCatch({
    # Use direction="auto" to handle both positive and negative correlations
    roc_obj <- roc(target, predictor, direction="auto")
    return(list(auc = as.numeric(auc(roc_obj)), direction = roc_obj$direction))
  }, error = function(e) {
    return(list(auc = NA, direction = NA))
  })
}

# Calculate correct AUC for each predictor
all_vars <- names(data)[names(data) != "Diagnosis"]
auc_results_list <- lapply(all_vars, function(var) calculate_correct_auc(var, data))

auc_results <- data.frame(
  Variable = all_vars,
  AUC = sapply(auc_results_list, function(x) x$auc),
  Direction = sapply(auc_results_list, function(x) x$direction),
  stringsAsFactors = FALSE
)

# Sort by AUC (higher is better)
auc_results <- auc_results[order(auc_results$AUC, decreasing = TRUE), ]
auc_results$AUC <- round(auc_results$AUC, 3)

print("Top predictors ranked by AUC (higher is better):")
print(head(auc_results, 15))

# Plot ROC curves for top predictors correctly
top_predictors <- head(auc_results$Variable, 5)
colors <- c("red", "blue", "green", "purple", "orange")

# Create a base plot
plot(x = c(0, 1), y = c(0, 1), type = "l", lty = 2, 
     xlab = "False Positive Rate", ylab = "True Positive Rate",
     main = "ROC Curves for Top Predictors")

# Add individual ROC curves
for (i in 1:length(top_predictors)) {
  var <- top_predictors[i]
  if(is.factor(data[[var]])) {
    predictor <- as.numeric(as.character(data[[var]]))
  } else {
    predictor <- data[[var]]
  }
  
  # Get target as binary
  target <- as.numeric(as.character(data$Diagnosis))
  
  tryCatch({
    # Get the ROC object with auto direction
    roc_obj <- roc(target, predictor, direction="auto")
    
    # Plot the ROC curve
    plot(roc_obj, add = TRUE, col = colors[i])
  }, error = function(e) {
    # Skip if error
    cat("Error with variable", var, ":", e$message, "\n")
  })
}

# Add a legend
legend("bottomright", legend = paste0(top_predictors, " (AUC: ", round(head(auc_results$AUC, 5), 2), ")"), 
       col = colors[1:length(top_predictors)], lwd = 2)

# Create a more detailed visualization with ggplot2
# First create a data frame to store ROC curve data for top predictors
roc_data <- data.frame()

for (i in 1:length(top_predictors)) {
  var <- top_predictors[i]
  if(is.factor(data[[var]])) {
    predictor <- as.numeric(as.character(data[[var]]))
  } else {
    predictor <- data[[var]]
  }
  
  # Get target as binary
  target <- as.numeric(as.character(data$Diagnosis))
  
  tryCatch({
    # Get the ROC object with auto direction
    roc_obj <- roc(target, predictor, direction="auto")
    
    # Extract coordinates
    coords <- coords(roc_obj, "all")
    
    # Add to data frame
    var_data <- data.frame(
      specificity = coords$specificity,
      sensitivity = coords$sensitivity,
      variable = var,
      auc = round(as.numeric(auc(roc_obj)), 3)
    )
    
    roc_data <- rbind(roc_data, var_data)
  }, error = function(e) {
    # Skip if error
    cat("Error with variable", var, ":", e$message, "\n")
  })
}

# Create the ggplot
ggplot(roc_data, aes(x = 1-specificity, y = sensitivity, color = variable)) +
  geom_line(size = 1) +
  geom_abline(slope = 1, intercept = 0, linetype = "dashed", color = "gray") +
  labs(
    title = "ROC Curves for Top Predictors",
    x = "False Positive Rate (1 - Specificity)",
    y = "True Positive Rate (Sensitivity)"
  ) +
  scale_color_manual(values = colors[1:length(top_predictors)], 
                     labels = paste0(unique(roc_data$variable), " (AUC: ", unique(roc_data$auc), ")")) +
  theme_minimal() +
  coord_equal() +
  theme(legend.title = element_blank(),
        legend.position = "bottom")






































# Function to bin continuous variables for MI calculation
discretize_var <- function(x, nbins = 10) {
  if (is.numeric(x)) {
    return(discretize(x, nbins = nbins))
  } else {
    return(as.numeric(x))
  }
}

# Prepare data for MI calculation
mi_data <- data
mi_data$Diagnosis <- as.numeric(as.character(mi_data$Diagnosis))

# Calculate MI for each variable with Diagnosis
mi_results <- data.frame(
  Variable = character(),
  MI = numeric(),
  NormalizedMI = numeric(),
  stringsAsFactors = FALSE
)

for (var in names(mi_data)[names(mi_data) != "Diagnosis"]) {
  tryCatch({
    # Discretize if numeric
    x_disc <- discretize_var(mi_data[[var]])
    y_disc <- mi_data$Diagnosis
    
    # Calculate mutual information
    mi_value <- mutinformation(x_disc, y_disc)
    
    # Calculate entropy for normalization
    h_x <- entropy(table(x_disc)/length(x_disc))
    h_y <- entropy(table(y_disc)/length(y_disc))
    
    # Normalized MI (ranges from 0 to 1)
    nmi <- 2 * mi_value / (h_x + h_y)
    
    # Add to results
    mi_results <- rbind(mi_results, data.frame(
      Variable = var,
      MI = mi_value,
      NormalizedMI = nmi,
      stringsAsFactors = FALSE
    ))
  }, error = function(e) {
    # Skip variables that cause errors
    cat("Error with variable", var, ":", e$message, "\n")
  })
}

# Sort by MI
mi_results <- mi_results[order(mi_results$MI, decreasing = TRUE), ]

# Print top variables by mutual information
print("Top predictors by Mutual Information:")
print(head(mi_results, 15))

# Plot mutual information
ggplot(head(mi_results, 15), aes(x = reorder(Variable, MI), y = MI)) +
  geom_bar(stat = "identity", fill = "darkblue") +
  coord_flip() +
  theme_minimal() +
  labs(title = "Top 15 Predictors by Mutual Information with Diagnosis",
       x = "Variables",
       y = "Mutual Information")

# Plot normalized mutual information
ggplot(head(mi_results, 15), aes(x = reorder(Variable, NormalizedMI), y = NormalizedMI)) +
  geom_bar(stat = "identity", fill = "darkred") +
  coord_flip() +
  theme_minimal() +
  labs(title = "Top 15 Predictors by Normalized Mutual Information with Diagnosis",
       x = "Variables",
       y = "Normalized Mutual Information (0-1)")

# Compare different feature selection methods
# Extract top features from each method

# Get top 10 features from each method
top_mi <- head(mi_results$Variable, 10)

# Use previously calculated results
top_rf <- head(importance_df[order(importance_df$MeanDecreaseGini, decreasing = TRUE), "Variable"], 10)
top_auc <- head(auc_results$Variable, 10)

# Find overlapping features
feature_comparison <- data.frame(
  Feature = unique(c(top_mi, top_rf, top_auc)),
  MI_Rank = rep(NA, length(unique(c(top_mi, top_rf, top_auc)))),
  RF_Rank = rep(NA, length(unique(c(top_mi, top_rf, top_auc)))),
  AUC_Rank = rep(NA, length(unique(c(top_mi, top_rf, top_auc))))
)

# Fill in rankings
for (i in 1:nrow(feature_comparison)) {
  feat <- feature_comparison$Feature[i]
  
  # MI rank
  mi_pos <- which(mi_results$Variable == feat)
  if (length(mi_pos) > 0) feature_comparison$MI_Rank[i] <- mi_pos
  
  # RF rank
  rf_pos <- which(importance_df$Variable == feat)
  if (length(rf_pos) > 0) feature_comparison$RF_Rank[i] <- rf_pos
  
  # AUC rank
  auc_pos <- which(auc_results$Variable == feat)
  if (length(auc_pos) > 0) feature_comparison$AUC_Rank[i] <- auc_pos
}

# Sort by average rank
feature_comparison$Avg_Rank <- rowMeans(feature_comparison[, c("MI_Rank", "RF_Rank", "AUC_Rank")], na.rm = TRUE)
feature_comparison <- feature_comparison[order(feature_comparison$Avg_Rank), ]

print("Feature rankings across different methods:")
print(head(feature_comparison, 15))

# Create a visualization of feature importance across methods
# Gather the ranking data for visualization
comparison_long <- feature_comparison %>%
  dplyr::select(Feature, MI_Rank, RF_Rank, AUC_Rank) %>%
  gather(key = "Method", value = "Rank", -Feature) %>%
  filter(!is.na(Rank)) %>%
  mutate(Method = gsub("_Rank", "", Method))

# Only keep top 15 features by average rank
top_features <- head(feature_comparison$Feature, 15)
comparison_long <- comparison_long[comparison_long$Feature %in% top_features, ]

# Invert rank for plotting (lower rank = higher bar)
max_rank <- max(comparison_long$Rank, na.rm = TRUE)
comparison_long$InvertedRank <- max_rank - comparison_long$Rank + 1

# Plot rankings side by side
ggplot(comparison_long, aes(x = reorder(Feature, -Rank), y = InvertedRank, fill = Method)) +
  geom_bar(stat = "identity", position = "dodge") +
  coord_flip() +
  scale_fill_brewer(palette = "Set1") +
  theme_minimal() +
  labs(title = "Feature Importance Rankings Across Methods",
       x = "Features",
       y = "Importance (Inverted Rank)",
       fill = "Method")

# Create a heatmap of feature rankings
# Convert to wide format for heatmap
heatmap_data <- comparison_long %>%
  dplyr::select(Feature, Method, Rank) %>%
  spread(key = Method, value = Rank)

# Replace NA with max rank + 1
heatmap_data[is.na(heatmap_data)] <- max_rank + 1

# Convert back to long for ggplot
heatmap_long <- heatmap_data %>%
  gather(key = "Method", value = "Rank", -Feature)

# Plot heatmap
ggplot(heatmap_long, aes(x = Method, y = reorder(Feature, -Rank), fill = Rank)) +
  geom_tile() +
  scale_fill_gradient(low = "red", high = "white", 
                      name = "Rank", 
                      breaks = c(1, 5, 10, 15, 20),
                      limits = c(1, max_rank + 1)) +
  theme_minimal() +
  labs(title = "Feature Importance Heatmap Across Methods",
       x = "Method",
       y = "Features") +
  theme(axis.text.x = element_text(angle = 0))

# Calculate joint mutual information for pairs of top variables
# This helps identify complementary variables

# Get top 6 variables by MI
top_vars <- head(mi_results$Variable, 6)
joint_mi <- data.frame(
  Var1 = character(),
  Var2 = character(),
  JointMI = numeric(),
  InteractionInfo = numeric(),
  stringsAsFactors = FALSE
)

for (i in 1:(length(top_vars)-1)) {
  for (j in (i+1):length(top_vars)) {
    var1 <- top_vars[i]
    var2 <- top_vars[j]
    
    # Discretize variables
    x1_disc <- discretize_var(mi_data[[var1]])
    x2_disc <- discretize_var(mi_data[[var2]])
    y_disc <- mi_data$Diagnosis
    
    # Calculate individual MIs
    mi1 <- mutinformation(x1_disc, y_disc)
    mi2 <- mutinformation(x2_disc, y_disc)
    
    # Calculate joint MI
    # Combine the two predictors
    combined <- paste(x1_disc, x2_disc, sep = "_")
    joint_mi_value <- mutinformation(combined, y_disc)
    
    # Calculate interaction information (positive = synergy, negative = redundancy)
    interaction_info <- joint_mi_value - (mi1 + mi2)
    
    # Add to results
    joint_mi <- rbind(joint_mi, data.frame(
      Var1 = var1,
      Var2 = var2,
      JointMI = joint_mi_value,
      InteractionInfo = interaction_info,
      stringsAsFactors = FALSE
    ))
  }
}

# Sort by interaction information
joint_mi <- joint_mi[order(joint_mi$InteractionInfo, decreasing = TRUE), ]

print("Variable pairs with highest synergy (positive interaction):")
print(head(joint_mi, 5))

print("Variable pairs with highest redundancy (negative interaction):")
print(tail(joint_mi, 5))

# Plot interaction information
ggplot(joint_mi, aes(x = paste(Var1, "&", Var2), y = InteractionInfo)) +
  geom_bar(stat = "identity", aes(fill = InteractionInfo > 0)) +
  coord_flip() +
  theme_minimal() +
  labs(title = "Interaction Information for Top Variable Pairs",
       x = "Variable Pairs",
       y = "Interaction Information",
       fill = "Synergy") +
  theme(axis.text.y = element_text(size = 8))

# Create a summary of findings
cat("\n====== MUTUAL INFORMATION ANALYSIS SUMMARY ======\n")
cat("Top 5 individual predictors by MI:\n")
print(head(mi_results, 5))

cat("\nMost synergistic pairs (complementary information):\n")
print(head(joint_mi, 3))

cat("\nMost redundant pairs (overlapping information):\n")
print(tail(joint_mi, 3))

cat("\nOptimal feature set based on combined methods:\n")
print(head(feature_comparison$Feature, 10))

























# Let's first identify the best predictors based on our previous analyses
# Using the feature comparison results or top features from MI, RF, and AUC

# For demonstration, I'll use what should be the top predictors
# If feature_comparison is available, uncomment this:
# top_predictors <- head(feature_comparison$Feature, 10)

# Otherwise, we'll manually specify based on previous results
top_predictors <- c("MMSE", "Disorientation", "FunctionalAssessment", 
                    "DifficultyCompletingTasks", "PersonalityChanges",
                    "Age", "SleepQuality", "Forgetfulness", "ADL", "PhysicalActivity")

# Prepare the model data
model_data <- data %>% 
  dplyr::select(all_of(c(top_predictors, "Diagnosis")))

# Handle any missing values if present
model_data <- na.omit(model_data)

# Split data into training and testing sets (70/30 split)
set.seed(123) # For reproducibility
train_index <- createDataPartition(model_data$Diagnosis, p = 0.7, list = FALSE)
train_data <- model_data[train_index, ]
test_data <- model_data[-train_index, ]

# Build the logistic regression model
log_model <- glm(Diagnosis ~ ., data = train_data, family = binomial)

# Model summary
summary_log <- summary(log_model)
print(summary_log)

# Make predictions on test data
test_probs <- predict(log_model, newdata = test_data, type = "response")

# Calculate ROC and find optimal threshold
roc_obj <- roc(test_data$Diagnosis, test_probs)
print(paste("AUC:", round(auc(roc_obj), 3)))

# Find the optimal threshold based on Youden's J statistic (maximizing sensitivity + specificity - 1)
coords_obj <- coords(roc_obj, "best", best.method = "youden")
optimal_threshold <- coords_obj$threshold
print(paste("Optimal threshold:", round(optimal_threshold, 3)))

# Apply the optimal threshold to get predictions
test_preds <- ifelse(test_probs > optimal_threshold, 1, 0)
test_preds <- factor(test_preds, levels = levels(test_data$Diagnosis))

# Calculate performance metrics
conf_matrix <- confusionMatrix(test_preds, test_data$Diagnosis, positive = "1")
print(conf_matrix)

# Plot ROC curve with threshold
plot(roc_obj, main = "ROC Curve for Logistic Regression Model")
points(coords_obj$specificity, coords_obj$sensitivity, pch = 19, col = "red")
text(coords_obj$specificity, coords_obj$sensitivity, 
     labels = paste("Threshold =", round(optimal_threshold, 2)), pos = 4)

# Let's also examine the predicted probabilities distribution
pred_df <- data.frame(
  Actual = test_data$Diagnosis,
  Probability = test_probs
)

# Plot probability distributions by actual class
ggplot(pred_df, aes(x = Probability, fill = Actual)) +
  geom_density(alpha = 0.5) +
  geom_vline(xintercept = optimal_threshold, linetype = "dashed", color = "red") +
  annotate("text", x = optimal_threshold + 0.05, y = 0.5, 
           label = paste("Threshold =", round(optimal_threshold, 2))) +
  labs(title = "Distribution of Predicted Probabilities by Actual Class",
       x = "Predicted Probability", 
       y = "Density") +
  theme_minimal()






















#https://shiring.github.io/machine_learning/2017/03/16/rf_plot_ggraph

# and log_model, roc_obj, and conf_matrix already exist from previous code

# Build a Random Forest model
set.seed(123) # For reproducibility
rf_model <- randomForest(
  Diagnosis ~ ., 
  data = train_data,
  ntree = 500,           # Number of trees
  mtry = sqrt(ncol(train_data) - 1), # Default mtry for classification
  importance = TRUE      # Calculate variable importance
)

# Print model summary
print(rf_model)

# Make predictions with the Random Forest model
rf_probs <- predict(rf_model, newdata = test_data, type = "prob")[,2]
rf_preds <- predict(rf_model, newdata = test_data, type = "class")

# Calculate ROC for Random Forest model
rf_roc <- roc(test_data$Diagnosis, rf_probs)
print(paste("Random Forest AUC:", round(auc(rf_roc), 3)))

# Calculate performance metrics for Random Forest model
rf_conf_matrix <- confusionMatrix(rf_preds, test_data$Diagnosis, positive = "1")
print(rf_conf_matrix)

# Plot variable importance
varImpPlot(rf_model, 
           main = "Random Forest - Variable Importance",
           n.var = min(20, ncol(train_data) - 1)) # Show top 20 variables or less

# Compare ROC curves of all models
plot(roc_obj, col = "blue", main = "Comparison of ROC Curves")
lines(rf_roc, col = "green")
if(exists("tree_roc")) {
  lines(tree_roc, col = "red")
  legend("bottomright", 
         legend = c(
           paste("Logistic Regression (AUC =", round(auc(roc_obj), 3), ")"),
           paste("Random Forest (AUC =", round(auc(rf_roc), 3), ")"),
           paste("Decision Tree (AUC =", round(auc(tree_roc), 3), ")")
         ),
         col = c("blue", "green", "red"), 
         lwd = 2)
} else {
  legend("bottomright", 
         legend = c(
           paste("Logistic Regression (AUC =", round(auc(roc_obj), 3), ")"),
           paste("Random Forest (AUC =", round(auc(rf_roc), 3), ")")
         ),
         col = c("blue", "green"), 
         lwd = 2)
}

# Create a comparison table of model performance
model_comparison <- data.frame(
  Model = c("Logistic Regression", "Random Forest"),
  AUC = c(round(auc(roc_obj), 3), round(auc(rf_roc), 3)),
  Accuracy = c(round(conf_matrix$overall["Accuracy"], 3), 
               round(rf_conf_matrix$overall["Accuracy"], 3)),
  Sensitivity = c(round(conf_matrix$byClass["Sensitivity"], 3), 
                  round(rf_conf_matrix$byClass["Sensitivity"], 3)),
  Specificity = c(round(conf_matrix$byClass["Specificity"], 3), 
                  round(rf_conf_matrix$byClass["Specificity"], 3)),
  PPV = c(round(conf_matrix$byClass["Pos Pred Value"], 3), 
          round(rf_conf_matrix$byClass["Pos Pred Value"], 3)),
  NPV = c(round(conf_matrix$byClass["Neg Pred Value"], 3), 
          round(rf_conf_matrix$byClass["Neg Pred Value"], 3))
)

# Add Decision Tree results if they exist
if(exists("tree_conf_matrix")) {
  tree_row <- data.frame(
    Model = "Decision Tree",
    AUC = round(auc(tree_roc), 3),
    Accuracy = round(tree_conf_matrix$overall["Accuracy"], 3),
    Sensitivity = round(tree_conf_matrix$byClass["Sensitivity"], 3),
    Specificity = round(tree_conf_matrix$byClass["Specificity"], 3),
    PPV = round(tree_conf_matrix$byClass["Pos Pred Value"], 3),
    NPV = round(tree_conf_matrix$byClass["Neg Pred Value"], 3)
  )
  model_comparison <- rbind(model_comparison, tree_row)
}

print(model_comparison)

# Extract feature importance from Random Forest
rf_importance <- importance(rf_model)
rf_importance_df <- data.frame(
  Feature = rownames(rf_importance),
  MeanDecreaseAccuracy = rf_importance[,"MeanDecreaseAccuracy"],
  MeanDecreaseGini = rf_importance[,"MeanDecreaseGini"],
  stringsAsFactors = FALSE
)
rf_importance_df <- rf_importance_df[order(rf_importance_df$MeanDecreaseGini, decreasing = TRUE),]

# Print feature importance 
print("Random Forest - Feature Importance:")
print(rf_importance_df)

# Compare feature importance across models
if(exists("log_importance") && exists("tree_importance")) {
  print("Feature Importance Comparison Across Models:")
  # Take top 10 features from each model to avoid excessive output
  top_features <- unique(c(
    head(log_importance$Feature, 10),
    head(rf_importance_df$Feature, 10),
    head(tree_importance$Feature, 10)
  ))
  
  importance_comparison <- data.frame(Feature = top_features)
  
  # Add logistic regression importance
  log_imp <- data.frame(Feature = log_importance$Feature, LogReg = log_importance$Importance)
  importance_comparison <- merge(importance_comparison, log_imp, by = "Feature", all.x = TRUE)
  
  # Add random forest importance (using Gini)
  rf_imp <- data.frame(Feature = rf_importance_df$Feature, RandomForest = rf_importance_df$MeanDecreaseGini)
  importance_comparison <- merge(importance_comparison, rf_imp, by = "Feature", all.x = TRUE)
  
  # Add tree importance
  tree_imp <- data.frame(Feature = tree_importance$Feature, DecisionTree = tree_importance$Importance)
  importance_comparison <- merge(importance_comparison, tree_imp, by = "Feature", all.x = TRUE)
  
  # Sort by random forest importance
  importance_comparison <- importance_comparison[order(importance_comparison$RandomForest, decreasing = TRUE),]
  print(importance_comparison)
}

# Out-of-bag error plot
plot(rf_model, main = "Random Forest - Out-of-Bag Error Rate")

# Evaluate model on different threshold values if needed
thresholds <- seq(0.1, 0.9, by = 0.1)
threshold_results <- data.frame(
  Threshold = thresholds,
  Accuracy = numeric(length(thresholds)),
  Sensitivity = numeric(length(thresholds)),
  Specificity = numeric(length(thresholds)),
  F1_Score = numeric(length(thresholds))
)

for(i in 1:length(thresholds)) {
  threshold <- thresholds[i]
  preds <- ifelse(rf_probs > threshold, "1", "0")
  preds <- factor(preds, levels = levels(test_data$Diagnosis))
  
  cm <- confusionMatrix(preds, test_data$Diagnosis, positive = "1")
  
  threshold_results$Accuracy[i] <- cm$overall["Accuracy"]
  threshold_results$Sensitivity[i] <- cm$byClass["Sensitivity"]
  threshold_results$Specificity[i] <- cm$byClass["Specificity"]
  threshold_results$F1_Score[i] <- cm$byClass["F1"]
}

print("Performance metrics at different threshold values:")
print(threshold_results)

# Plot threshold analysis
par(mfrow = c(2, 2))
plot(threshold_results$Threshold, threshold_results$Accuracy, 
     type = "b", col = "blue", main = "Accuracy vs Threshold", 
     xlab = "Threshold", ylab = "Accuracy")

plot(threshold_results$Threshold, threshold_results$Sensitivity, 
     type = "b", col = "red", main = "Sensitivity vs Threshold",
     xlab = "Threshold", ylab = "Sensitivity")

plot(threshold_results$Threshold, threshold_results$Specificity, 
     type = "b", col = "green", main = "Specificity vs Threshold",
     xlab = "Threshold", ylab = "Specificity")

plot(threshold_results$Threshold, threshold_results$F1_Score, 
     type = "b", col = "purple", main = "F1 Score vs Threshold",
     xlab = "Threshold", ylab = "F1 Score")
par(mfrow = c(1, 1))

# Optimal threshold based on balanced accuracy
optimal_idx <- which.max((threshold_results$Sensitivity + threshold_results$Specificity) / 2)
optimal_threshold <- threshold_results$Threshold[optimal_idx]
print(paste("Optimal threshold for balanced performance:", optimal_threshold))







getTree(rf_model, 1, labelVar=TRUE)




























# Build a simple decision tree
# Decision trees automatically select important variables
tree_model <- rpart(Diagnosis ~ ., data = train_data, 
                    method = "class",
                    control = rpart.control(maxdepth = 4, # Limit depth for simplicity
                                            cp = 0.01)) # Complexity parameter
# Plot the decision tree
rpart.plot(tree_model, extra = 104, # Show counts and probabilities
           box.palette = "RdBu", # Color scheme
           fallen.leaves = TRUE, # Align leaf nodes
           main = "Decision Tree for Diagnosis")
# Print the decision rules
print(tree_model)
# Make predictions with the tree model
tree_probs <- predict(tree_model, newdata = test_data, type = "prob")[,2]
tree_preds <- predict(tree_model, newdata = test_data, type = "class")
# Calculate ROC for tree model
tree_roc <- roc(test_data$Diagnosis, tree_probs)
print(paste("Decision Tree AUC:", round(auc(tree_roc), 3)))
# Calculate performance metrics for tree model
tree_conf_matrix <- confusionMatrix(tree_preds, test_data$Diagnosis, positive = "1")
print(tree_conf_matrix)
# Compare ROC curves of both models
plot(roc_obj, col = "blue", main = "Comparison of ROC Curves")
lines(tree_roc, col = "red")
legend("bottomright", legend = c(paste("Logistic Regression (AUC =", round(auc(roc_obj), 3), ")"),
                                 paste("Decision Tree (AUC =", round(auc(tree_roc), 3), ")")),
       col = c("blue", "red"), lwd = 2)
# Create a comparison table of model performance
model_comparison <- data.frame(
  Model = c("Logistic Regression", "Decision Tree"),
  AUC = c(round(auc(roc_obj), 3), round(auc(tree_roc), 3)),
  Accuracy = c(round(conf_matrix$overall["Accuracy"], 3), 
               round(tree_conf_matrix$overall["Accuracy"], 3)),
  Sensitivity = c(round(conf_matrix$byClass["Sensitivity"], 3), 
                  round(tree_conf_matrix$byClass["Sensitivity"], 3)),
  Specificity = c(round(conf_matrix$byClass["Specificity"], 3), 
                  round(tree_conf_matrix$byClass["Specificity"], 3)),
  PPV = c(round(conf_matrix$byClass["Pos Pred Value"], 3), 
          round(tree_conf_matrix$byClass["Pos Pred Value"], 3)),
  NPV = c(round(conf_matrix$byClass["Neg Pred Value"], 3), 
          round(tree_conf_matrix$byClass["Neg Pred Value"], 3))
)
print(model_comparison)
# Print feature importance comparisons
print("Logistic Regression - Feature Importance:")
print(log_importance)
# Create a prediction function based on the decision tree rules
print("Simplified decision rules for diagnosis prediction:")
# This will print a user-friendly version of the main decision rules
tree_rules <- rpart.rules(tree_model, cover = TRUE, style = "tallw")
print(tree_rules)

























# Pair plot of a subset of variables (can be computationally intensive for many variables)
if (ncol(data) > 3) {
  ggpairs(data %>% dplyr::select(Age, Gender, Ethnicity, MMSE, FunctionalAssessment, Diagnosis),
          aes(color = factor(Diagnosis)))
}

