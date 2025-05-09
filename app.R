library(tidyverse)
library(shiny)
library(gbm)
library(caret)
library(pROC)
library(ggplot2)
library(scales)
library(ggtext)



predictors <- c(
  "FunctionalAssessment",
  "ADL",
  "MemoryComplaints",
  "BehavioralProblems",
  "MMSE",
  "CholesterolHDL",
  "SleepQuality"
)

# a rainbow of the same length, named by predictor
pal_global <- setNames(rainbow(length(predictors)), predictors)



data_raw <- read.csv("alzheimers_disease_data.csv")
data <- data_raw %>% dplyr::select(-DoctorInCharge, -PatientID)

# turn 0/1 columns into factors before training
data$Diagnosis <- factor(
  data$Diagnosis,
  levels = c(0, 1),
  labels = c("No", "Yes")
)

data$MemoryComplaints   <- factor(
  data$MemoryComplaints,
  levels = c(0,1),
  labels = c("No","Yes")
)

data$BehavioralProblems <- factor(
  data$BehavioralProblems,
  levels = c(0,1),
  labels = c("No","Yes")
)

set.seed(65778275)
train_idx  <- createDataPartition(data$Diagnosis, 
                                  p = 0.7, 
                                  list = FALSE)
train_data <- data[train_idx, ]
test_data  <- data[-train_idx, ]

# Train models
set.seed(65778275)
# GLM (no resampling)
glm_model <- train(
  Diagnosis ~ FunctionalAssessment + ADL + MemoryComplaints +
    BehavioralProblems + MMSE + CholesterolHDL + SleepQuality,
  data     = train_data,
  method   = "glm",
  family   = binomial(),
  trControl = trainControl(method = "none")  # no CV here
)

set.seed(65778275)
# Set up trainControl for classification + ROC + saved preds
gbm_ctrl <- trainControl(
  method            = "cv",            # 10-fold CV
  number            = 10,
  classProbs        = TRUE,            # save P(Yes), P(No)
  summaryFunction   = twoClassSummary,
  savePredictions = "final"
)

# GBM (with ROC tuning & saved predictions)
set.seed(65778275)
gbm_model <- train(
  Diagnosis ~ FunctionalAssessment + ADL + MemoryComplaints +
    BehavioralProblems + MMSE + CholesterolHDL + SleepQuality,
  data      = train_data,
  method    = "gbm",
  metric    = "ROC",
  trControl = gbm_ctrl,
  verbose   = FALSE
)

best  <- gbm_model$bestTune
# subset 'results' to the row that matches your bestTune
cv_auc <- gbm_model$results %>%
  dplyr::filter(
    n.trees          == best$n.trees,
    interaction.depth== best$interaction.depth,
    shrinkage        == best$shrinkage,
    n.minobsinnode   == best$n.minobsinnode
  ) %>%
  pull(ROC) %>%
  round(3)


# EDA
# Pre-calculate EDA elements
outcome_table <- table(data$Diagnosis)
outcome_prop <- prop.table(outcome_table)


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


numeric_data <- bind_cols(numeric_attrs, Diagnosis = data$Diagnosis)
numeric_long <- numeric_data %>%
  pivot_longer(-Diagnosis, names_to = "variable", values_to = "value")
numeric_densities <- ggplot(numeric_long, aes(x = value, fill = Diagnosis)) +
  geom_density(alpha = 0.5) +
  facet_wrap(~ variable, scales = "free", ncol = 3) +
  labs(title = "Density Plots of Numeric Variables by Diagnosis",
       x = "Value", y = "Density") +
  theme_minimal()

factor_data <- data %>%
  select_if(is.factor)

factor_data <- factor_data %>%
  dplyr::select(-EducationLevel, -Ethnicity)

factor_distributions <- factor_data %>%
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

# Predictor selection results
predictor_selection_results <- tibble::tribble(
  ~Variable,             ~Test,   ~Statistic, ~p_value, ~Significant,
  "MMSE",                 "t‑test",     12.0,  0,       "Yes",
  "FunctionalAssessment", "t‑test",     18.6,  0,       "Yes",
  "ADL",                  "t‑test",     16.5,  0,       "Yes",
  "MemoryComplaints",     "χ²",        201.,  0,       "Yes",
  "BehavioralProblems",   "χ²",        107.,  0,       "Yes",
  "SleepQuality",         "t‑test",      2.63, 0.00867, "Yes",
  "CholesterolHDL",       "t‑test",     -1.97, 0.0490,  "Yes",
  "Ethnicity",            "χ²",          6.30, 0.0978,  "No",
  "Hypertension",         "χ²",          2.44, 0.118,   "No",
  "CholesterolLDL",       "t‑test",      1.48, 0.138,   "No",
  "Age",                  "t‑test",      1.12, 0.262,   "No",
  "Gender",               "χ²",          0.11, 0.737,   "No",
  "YearsOfEducation",     "t‑test",      0.88, 0.378,   "No",
  "MaritalStatus",        "χ²",          1.56, 0.816,   "No",
  "LivingAlone",          "χ²",          0.01, 0.920,   "No",
  "Smoking",              "χ²",          0.23, 0.630,   "No",
  "AlcoholConsumption",   "χ²",          1.08, 0.300,   "No",
  "HeadInjury",           "χ²",          0.00, 0.990,   "No",
  "Diabetes",             "χ²",          0.01, 0.910,   "No",
  "HeartDisease",         "χ²",          0.02, 0.880,   "No",
  "Stroke",               "χ²",          0.04, 0.840,   "No",
  "Depression",           "χ²",          0.00, 0.960,   "No",
  "Anxiety",              "χ²",          0.01, 0.930,   "No",
  "ParkinsonsDisease",    "χ²",          0.00, 0.970,   "No",
  "MultipleSclerosis",    "χ²",          0.00, 1.00,    "No",
  "Epilepsy",             "χ²",          0.00, 1.00,    "No",
  "Cancer",               "χ²",          0.00, 1.00,    "No",
  "ThyroidProblems",      "χ²",          0.00, 1.00,    "No",
  "VitaminB12Deficiency", "χ²",          0.00, 1.00,    "No",
  "SleepApnea",           "χ²",          0.00, 1.00,    "No",
  "Falls",                "χ²",          0.00, 1.00,    "No",
  "FamilyHistoryAlzheimers", "χ²",          1.58, 0.209,   "No"
)


# Load the trained models (adjust paths if needed)
glm_model_exp <- readRDS("glm_model.rds")
gbm_model_exp <- readRDS("gbm_model.rds");




# ------------------------------------------------------------------------------
## Functions

# Function to make some nice tooltips
make_tooltip <- function(items) {
  # items: a named character vector or list
  # returns one string with each “name: value” on its own line
  entries <- sprintf(
    "%s: %s",
    names(items),
    unlist(items, 
    use.names = FALSE)
  )
  paste(entries, collapse = "\n")
}

# Function for variable‐importance plot for any caret::train model
plotImportance <- function(model) {
  # 1) build the raw df
  if (model$method == "glm") {
    coefs <- coef(model$finalModel)
    df    <- data.frame(
      Variable   = names(coefs)[-1],
      Importance = abs(as.numeric(coefs[-1])),
      stringsAsFactors = FALSE
    )
  } else if (model$method == "gbm") {
    imp <- summary(model, plotit = FALSE)
    df  <- data.frame(
      Variable   = imp$var,       # e.g. "MemoryComplaintsYes"
      Importance = imp$rel.inf,
      stringsAsFactors = FALSE
    )
  } else {
    stop("Unsupported model for plotImportance")
  }
  
  # 2) strip off any trailing "Yes"/"No" so we get the base var
  df$BaseVar <- sub("(Yes|No)$", "", df$Variable)
  
  # 3) plot – fill by BaseVar, using our global palette
  ggplot(df, aes(
    x    = reorder(Variable, Importance),
    y    = Importance,
    fill = BaseVar
  )) +
    geom_col() +
    coord_flip() +
    scale_fill_manual(
      values = pal_global,
      guide  = "none"
    ) +
    labs(
      title = if (model$method=="glm")
        "GLM Predictor Importance"
      else
        "GBM Predictor Importance",
      x = "Predictor",
      y = "Importance"
    ) +
    theme_minimal(base_size = 14) +
    theme(plot.title = element_text(hjust = 0.5))
}

# Function for ROC‐curve plot for any caret::train model with saved preds
plotROC <- function(model) {
  # pull out the “Yes” probs & true obs from the cross-val preds (if available)
  if (!is.null(model$pred)) {
    tuneName <- names(model$bestTune)[1]
    bestVal  <- model$bestTune[[1]]
    preds    <- model$pred[ model$pred[[tuneName]] == bestVal, ]
    roc_obj  <- pROC::roc(preds$obs, preds$Yes)
  } else {
    td      <- model$trainingData
    actual  <- td$.outcome
    X       <- td[, setdiff(names(td), ".outcome"), drop = FALSE]
    if (model$method == "glm") {
      probs <- predict(model, X, type = "prob")[, "Yes"]
    } else if (model$method == "gbm") {
      probs <- predict(
        model$finalModel,
        newdata = X,
        n.trees = model$bestTune$n.trees,
        type    = "response"
      )
    }
    roc_obj <- pROC::roc(actual, probs)
  }
  
  # assemble data.frame
  roc_df <- data.frame(
    fpr = 1 - roc_obj$specificities,
    tpr =    roc_obj$sensitivities
  )
  auc_val <- pROC::auc(roc_obj)
  
  # choose color by model type
  curve_col <- if (model$method == "glm") "royalblue" else "brown"
  
  ggplot(roc_df, aes(x = fpr, y = tpr)) +
    geom_line(size = 1, color = curve_col) +
    geom_abline(linetype = "dashed", color = "gray") +
    annotate(
      "text",
      x     = 0.75, y     = 0.25,
      label = paste0("AUC = ", round(auc_val, 3)),
      color = curve_col,
      size  = 4
    ) +
    labs(
      x = "False Positive Rate (1 − Specificity)",
      y = "True Positive Rate (Sensitivity)",
      title = if (model$method=="glm") 
        "GLM ROC Curve" 
      else 
        "GBM ROC Curve"
    ) +
    theme_minimal() +
    theme(plot.title = element_text(hjust = 0.5))
}

plotROCOnData <- function(model, newdata) {
  actual <- as.character(newdata$Diagnosis)
  
  # 2a) predict probabilities
  if (model$method == "glm") {
    probs <- predict(model$finalModel, newdata, type = "response")
  } else if (model$method == "gbm") {
    probs <- caret::predict.train(model, newdata, type = "prob")[, "Yes"]
  }
  
  # 2b) build the ROC object
  roc_obj <- pROC::roc(actual, probs)
  roc_df  <- data.frame(
    fpr = 1 - roc_obj$specificities,
    tpr =    roc_obj$sensitivities
  )
  auc_val <- pROC::auc(roc_obj)
  curve_col <- if (model$method == "glm") "royalblue" else "firebrick"
  
  # 2c) ggplot it with annotation
  ggplot(roc_df, aes(x = fpr, y = tpr)) +
    geom_line(size = 1, color = curve_col) +
    geom_abline(lty = 2, color = "grey50") +
    annotate("text", x = 0.75, y = 0.25,
             label = paste0("AUC = ", round(auc_val, 3)),
             color = curve_col, size = 4) +
    labs(
      x     = "False Positive Rate (1 − Specificity)",
      y     = "True Positive Rate (Sensitivity)",
      title = if (model$method == "glm")
        "GLM ROC Curve (Test)"
      else
        "GBM ROC Curve (Test)"
    ) +
    theme_minimal() +
    theme(plot.title = element_text(hjust = 0.5))
}

# Function to compute & print a confusion matrix from a caret model
getConfusion <- function(model, newdata, threshold = 0.5) {
  actual <- as.character(newdata$Diagnosis)
  # get P(Yes):
  if (model$method == "glm") {
    probs <- predict(model, newdata, type = "prob")[, "Yes"]
  } else {
    probs <- predict(
      model$finalModel,
      newdata = newdata[, setdiff(names(newdata), "Diagnosis")],
      n.trees = model$bestTune$n.trees,
      type    = "response"
    )
  }
  pred_chr <- ifelse(probs > threshold, "Yes", "No")
  cm_tbl    <- table(Predicted = pred_chr, Actual = actual)
  caret::confusionMatrix(cm_tbl)
}

getConfusionOnData <- function(model, newdata, threshold = 0.5) {
  actual <- as.character(newdata$Diagnosis)
  
  # strip off the outcome column before predicting
  df <- newdata[, setdiff(names(newdata), "Diagnosis")]
  # get predicted P(Yes) for *any* caret model
  probs <- caret::predict.train(model, df, type = "prob")[, "Yes"]
  
  # 1b) threshold into classes
  preds <- ifelse(probs > threshold, "Yes", "No")
  
  # 1c) build a 2×2 table and hand to caret
  cm_tbl <- table(Predicted = preds, Actual = actual)
  caret::confusionMatrix(cm_tbl)
}

plotConfusionGrid <- function(model, newdata, threshold = 0.5) {
  cm_obj <- getConfusionOnData(model, newdata, threshold)
  df     <- as.data.frame(cm_obj$table, stringsAsFactors = FALSE)
  names(df) <- c("Prediction", "Reference", "Count")
  df$Prediction <- factor(df$Prediction, levels = c("No","Yes"))
  df$Reference  <- factor(df$Reference,  levels = c("No","Yes"))
  ggplot(df, aes(Prediction, Reference, fill = Count)) +
    geom_tile(color="black", size=0.6) +
    geom_text(aes(label = Count), size = 6) +
    scale_fill_gradient(low="pink", high="purple4") +
    labs(x="Predicted", y="Actual", title="Confusion Matrix (Test Set)") +
    theme_minimal(base_size = 14) +
    theme(plot.title = element_text(hjust=0.5),
          legend.position="none")
}







## LISTS FOR MANUAL TESTS FOR KEY PREDICTORS
# List of the 10 ADL items and their simple descriptions based on Barthel Index
# https://onlinelibrary.wiley.com/doi/10.1111/j.1748-3743.2007.00074.x -> Table 1.
adl_items <- list(
  Bathing = "bathing self without help",
  Dressing = "getting clothes on/off unassisted",
  Toileting = "using toilet and cleaning self",
  Transferring = "moving in/out of bed or chair",
  Continence = "controlling bladder/bowel",
  Feeding = "eating without assistance",
  Grooming = "personal hygiene independently",
  Walking = "ambulating without aid",
  Stairs = "climbing stairs safely",
  WheelchairUse = "operating wheelchair unassisted"
)

# There are multiple studies on FunctionalAssessment. However none really use scoring in 0-10 range. We decided to use: Pfeffer’s Functional Activities Questionnaire (FAQ). Although this scoring assumes values in range 0-30, it can be simplified to just 0-10 giving us good metrcic for quick display of this test. Scales like: Lawton–Brody Instrumental Activities of Daily Living (IADL) Scale couldn't be used since they assume ranges in 0-8. There we suggestions on adding 2 more activities like Mobility and SelfCare but we found them pointless in the bigger picture of this dataset.
fs_items <- list(
  Finance = "Writing checks, paying bills, balancing checkbook",
  Paperwork = "Assembling tax records, business affairs, or papers",
  Shopping = "Shopping alone for clothes, household necessities, or groceries",
  Leisure = "Playing a game of skill or working on a hobby",
  KitchenUse = "Heating water, making a cup of coffee, turning off stove after use",
  MealPrep = "Preparing a balanced meal",
  CurrentEvents = "Keeping track of current events",
  MediaComprehension = "Paying attention to, understanding, discussing TV, books, or magazines",
  Memory = "Remembering appointments, family occasions, holidays, medications",
  Transportation = "Traveling out of neighborhood, driving, arranging to take buses"
)

# For MMSE we used this article: https://doi.org/10.1007/978-3-319-57111-9_199.
# Define the MMSE domains and their quick scoring guide
mmse_items <- list(
  Orientation = "0–10 points: date, day, season, location (country, city, building, floor, etc.)",
  Registration = "0–3 points: repeat three named objects (e.g. “apple, penny, table”)",
  AttentionCalculation = "0–5 points: serial 7’s subtraction (count from 100 down by 7s) or spell Tomato backwards",
  Recall = "0–3 points: recall the three meassures from ADL list",
  LanguagePraxis = "0–9 points: naming, 3-stage command, reading & obeying written instruction, writing a sentence, copying intersecting pentagons"
)



# Define UI for application
ui <- fluidPage(
  titlePanel("Alzheimer's Disease Prediction"),
  tabsetPanel(type = "tabs",
    # ——————————————————————————————
    # Tab 1: Exploratory Data Analysis
    tabPanel(
      "EDA Findings",
      h3("Exploratory Data Analysis Results"),
      h4("Histograms of Numeric Attributes"),
      plotOutput("numeric_densities"),
      h4("Distributions of Categorical Attributes"),
      plotOutput("factor_distributions"),
      h4("Key Findings"),
      verbatimTextOutput("eda_findings"),
      h4("Predictor Selection"),
      tableOutput("predictor_selection_table")
    ),          
              
    tabPanel("Predictions",          
      sidebarLayout(
        sidebarPanel(
          selectInput(
            "model_type", "Select Model:",
            choices = c(
              "Logistic Regression (GLM)", 
              "Gradient Boosting Machine (GBM)")
            ),
          hr(), 
          h4("Patient Data Input"),
          h5("Hower over each for more info"),
          numericInput(
            "FunctionalAssessment", 
            label = tags$span(
              style = paste("color:", pal_global["FunctionalAssessment"], "; font-weight:bold;"),
              "FunctionalAssessment (0–10)",
              title = paste(
                "Add 1 for each activity where severe assistance is needed:",
                make_tooltip(fs_items),
                sep = "\n\n"
              )
            ), 
            value = 0, min = 0, max = 10
          ),
          numericInput(
            "ADL",
            label = tags$span(
              style = paste("color:", pal_global["ADL"], "; font-weight:bold;"),
              "ADL - Activities of Daily Living (0–10)",
              title = paste(
                "Add 1 for each activity where assistance is needed:",
                make_tooltip(adl_items),
                sep = "\n\n"
              )
            ), 
            value = 0, min = 0, max = 10
          ),
          tags$div(
            style = paste("color:", pal_global["MemoryComplaints"], "; font-weight:bold;"),
            title = "Has the patient or caregiver noticed any memory lapses?",
            selectInput(
              "MemoryComplaints",
              "Memory Complaints",
              choices = c("No", "Yes")
            )
          ),
          
          tags$div(
            style = paste("color:", pal_global["BehavioralProblems"], "; font-weight:bold;"),
            title = "Any new or worsening behavioral symptoms (agitation, apathy, etc.)?",
            selectInput(
              "BehavioralProblems",
              "Behavioral Problems",
              choices = c("No", "Yes")
            )
          ),
          numericInput(
            "MMSE", 
            label = tags$span(
              style = paste("color:", pal_global["MMSE"], "; font-weight:bold;"),
              "MSE - Mini-Mental State Exam (0–30)",
              title = paste("Substract 1 from 30 for each failed task:",
              make_tooltip(mmse_items),
              sep = "\n\n"
              )
            ), 
            value = 0, min = 0, max = 30
          ),
          numericInput(
            "CholesterolHDL", 
            label   = tags$span(
              style = paste("color:", pal_global["CholesterolHDL"], "; font-weight:bold;"),
              "HDL Cholesterol (mg/dL)",
              title = "HDL (good) cholesterol helps clear plaque from your arteries. Higher for women due to effects of estrogen on lipid metabolism - process of synthesizing, breaking down, and storing fat in cells for energy storage\n\
    Aim for ≥40 mg/dL (men) or ≥50 mg/dL (women); 60+ mg/dL is optimal."
            ),
            value = 20, min = 20, max = 100
          ),
          # We did not find any studies with index in ranges 4-10. So we decided to interpret this number as average daily sleep time from 4-10 hours.
          numericInput(
            "SleepQuality", 
            label = tags$span(
              style = paste("color:", pal_global["SleepQuality"], "; font-weight:bold;"),
              "Sleep Quality (4–10)",
              title = "Average number of hours slept"
            ), 
            value = 4, min = 4, max = 10
          ),
          actionButton("predict", "Predict"),
          hr(),
          h4("Model Statistics"),
          verbatimTextOutput("model_stats") # To display model statistics
        ),
        mainPanel(
          h3("Prediction Results"),
          verbatimTextOutput("prediction"),
          conditionalPanel(
            condition = "input.model_type == 'Logistic Regression (GLM)'",
            #h4("GLM Predictor Importance"),
            #plotOutput("glm_importance_plot"),
            #hr(),
            h4("ROC Curve (GLM)"),
            plotOutput("glm_roc_plot"),
            hr(),
            h4("Confusion Matrix (Training)"),
            #verbatimTextOutput("glm_confusion"),
            plotOutput("glm_confusion_counts")
          ),
          
          conditionalPanel(
            condition = "input.model_type == 'Gradient Boosting Machine (GBM)'",
            h4("GBM Predictor Importance"),
            plotOutput("gbm_importance_plot"),
            hr(),
            h4("ROC Curve (GBM)"),
            plotOutput("gbm_roc_plot"),
            hr(),
            h4("Confusion Matrix (Training)"),
            #verbatimTextOutput("gbm_confusion"),
            plotOutput("gbm_confusion_counts")
          )
        )
      )
    )
  )
)

# Define server logic
server <- function(input, output) {
  
  # Predictions Tab
  #-----------------------------------------------------------------------------
  # Display Model Statistics
  output$model_stats <- renderText({
    # — GLM metrics on training
    glm_raw       <- glm_model$finalModel
    glm_train_aic <- round(AIC(glm_raw), 2)
    glm_null_dev  <- round(glm_raw$null.deviance, 2)
    glm_resid_dev <- round(glm_raw$deviance, 2)
    
    glm_info <- paste(
      "Logistic Regression (GLM):",
      sprintf("\n  AIC: %s",          glm_train_aic),
      sprintf("\n  Null Deviance: %s", glm_null_dev),
      sprintf("\n  Residual Deviance: %s", glm_resid_dev)
    )
    
    # — GBM metrics (optimal trees & test AUC)
    opt_trees      <- gbm_model$bestTune$n.trees
    
    # pull out only the preds for the chosen tuning parameter
    tune_names <- names(gbm_model$bestTune)
    is_best    <- Reduce(`&`, lapply(tune_names, function(nm) {
      gbm_model$pred[[nm]] == gbm_model$bestTune[[nm]]
    }))
    cv_preds   <- gbm_model$pred[is_best, ]
    
    
    gbm_info <- paste(
      "Gradient Boosting Machine (GBM):",
      sprintf("\n  Optimal Trees: %d", opt_trees),
      sprintf("\n  CV AUC:  %0.3f", cv_auc)
    )
    
    paste(glm_info, "\n\n", gbm_info)
  })
  
  
  preds <- eventReactive(input$predict, {
    df <- data.frame(
      FunctionalAssessment = input$FunctionalAssessment,
      ADL = input$ADL,
      MemoryComplaints = factor(
        input$MemoryComplaints, 
        levels = c("No","Yes")
      ),
      BehavioralProblems = factor(
        input$BehavioralProblems, 
        levels = c("No","Yes")
      ),
      MMSE = input$MMSE,
      CholesterolHDL = input$CholesterolHDL,
      SleepQuality = input$SleepQuality
    )
    
    if (input$model_type == "Logistic Regression (GLM)") {
      prob <- predict(glm_model, df, type = "prob")[, "Yes"]
      thresh <- 0.5
    } else {
      prob <- predict(gbm_model, df, type = "prob")[, "Yes"]
      thresh <- 0.3567714 
    }
    list(prob = prob, class = ifelse(prob > thresh, "AD", "No AD"))
  })
  
  output$prediction <- renderPrint({
    p <- preds()
    cat("Prob:", round(p$prob,3), "\nPrediction:", p$class)
  })
  
  # GLM part
  #output$glm_importance_plot <- renderPlot({
  #  req(input$model_type == "Logistic Regression (GLM)")
  #  plotImportance(glm_model)
  #})
  output$glm_roc_plot <- renderPlot({
    req(input$model_type == "Logistic Regression (GLM)")
    plotROC(glm_model)
  })
  #output$glm_confusion <- renderPrint({
  #  getConfusion(glm_model, test_data, threshold = 0.5)
  #})
  output$glm_confusion_counts <- renderPlot({
    plotConfusionGrid(glm_model, test_data, threshold = 0.5)
  })
  
  
  # GBM part
  output$gbm_importance_plot <- renderPlot({
    req(input$model_type == "Gradient Boosting Machine (GBM)")
    plotImportance(gbm_model)
  })
  output$gbm_roc_plot <- renderPlot({
    req(input$model_type == "Gradient Boosting Machine (GBM)")
    plotROCOnData(gbm_model, test_data)
  })
  #output$gbm_confusion <- renderPrint({
  #  getConfusionOnData(gbm_model, test_data, threshold = 0.3567714)
  #})
  output$gbm_confusion_counts <- renderPlot({
    plotConfusionGrid(gbm_model, test_data, threshold = 0.3567714)
  })
  
  
  
  # EDA Tab
  #-----------------------------------------------------------------------------
  output$outcome_balance <- renderText({
    paste(
      "Outcome Class Counts:\n",
      paste(names(outcome_table), outcome_table, sep = ": ", collapse = "\n"),
      "\n\nOutcome Class Proportions:\n",
      paste(
        names(outcome_prop), 
        round(outcome_prop * 100, 2), 
        sep = ": ", 
        collapse = "\n")
    )
  })
  
  output$numeric_densities <- renderPlot({
    numeric_densities
  })
  
  output$factor_distributions <- renderPlot({
    factor_distributions
  })
  
  output$eda_findings <- renderText({
    "- Observations: 2149\n- No missing values in any column -> no imputation needed\n- Slight imbalance (65% negative, 35% positive)\n- Numeric attributes - None is normally distributed, no outlier deletion\n- No significant correlation in dataset"
  })
  
  output$predictor_selection_table <- renderTable({
    predictor_selection_results %>%
      filter(Significant == "Yes") %>%
      dplyr::select(Variable, Test, p_value) %>%
      mutate(p_value = format.pval(p_value, digits = 3)) %>%
      head(12)
  })
  
}



# Run the application
shinyApp(ui = ui, server = server)


