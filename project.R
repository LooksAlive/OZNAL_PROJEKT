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

# Load the dataset (already done in your code)
data <- read.csv("alzheimers_disease_data.csv")

# Drop the 'DoctorInCharge' column
data <- data %>% select(-DoctorInCharge)
data <- data[-data$PatientID]
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
  if (is.numeric(data[[col_name]]) && all(unique(data[[col_name]]) %in% c(0, 1))) {
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
ggplot(data, aes(x = Age, y = Gender, color = Diagnosis)) +
  geom_point() +
  labs(title = "Gender & Age vs. Diagnosis", x = "Age", y = "Gender") +
  theme_minimal() +
  scale_y_continuous(breaks = c(0, 1), labels = c("Male", "Female")) +
  scale_color_discrete(labels = c("No Alzheimer's", "Alzheimer's"))

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












































# Pair plot of a subset of variables (can be computationally intensive for many variables)
if (ncol(data) > 3) {
  ggpairs(data %>% select(Age, Gender, Ethnicity, MMSE, FunctionalAssessment, Diagnosis),
          aes(color = factor(Diagnosis)))
}