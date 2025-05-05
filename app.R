library(shiny)
library(gbm)
library(caret)
library(pROC)
library(ggplot2)

# Load the trained models (adjust paths if needed)
glm_model <- readRDS("glm_model.rds")
gbm_model <- readRDS("gbm_model.rds")

# Define UI for application
ui <- fluidPage(
  titlePanel("Alzheimer's Disease Prediction"),
  sidebarLayout(
    sidebarPanel(
      selectInput("model_type", "Select Model:",
                  choices = c("Logistic Regression (GLM)", "Gradient Boosting Machine (GBM)")),
      hr(),
      h4("Patient Data Input"),
      numericInput("FunctionalAssessment", "Functional Assessment", value = 8, min = 0, max = 18),
      numericInput("ADL", "ADL", value = 15, min = 0, max = 24),
      selectInput("MemoryComplaints", "Memory Complaints", choices = c("0", "1")),
      selectInput("BehavioralProblems", "Behavioral Problems", choices = c("0", "1")),
      numericInput("MMSE", "MMSE", value = 25, min = 0, max = 30),
      numericInput("CholesterolHDL", "CholesterolHDL", value = 50, min = 20, max = 100),
      numericInput("SleepQuality", "SleepQuality", value = 7, min = 1, max = 10),
      actionButton("predict", "Predict"),
      hr(),
      h4("Model Statistics"),
      verbatimTextOutput("model_stats") # To display model statistics
    ),
    mainPanel(
      h3("Prediction Results"),
      verbatimTextOutput("prediction"),
      conditionalPanel(
        condition = "input.model_type == 'Gradient Boosting Machine (GBM)'",
        h4("GBM Predictor Importance"),
        plotOutput("gbm_importance_plot")
      )
    )
  )
)

# Define server logic
server <- function(input, output) {
  
  # Display Model Statistics
  output$model_stats <- renderText({
    glm_info <- paste("Logistic Regression (GLM):",
                      "\n  AIC:", round(AIC(glm_model), 2),
                      "\n  Null Deviance:", round(glm_model$null.deviance, 2),
                      "\n  Residual Deviance:", round(glm_model$deviance, 2))
    
    gbm_info <- paste("Gradient Boosting Machine (GBM):",
                      "\n  Optimal Trees:", gbm_model$bestTune$n.trees,
                      "\n  AUC (Cross-Validation):", round(max(gbm_model$results$ROC), 3)) # Assuming ROC was the metric
    
    paste(glm_info, "\n\n", gbm_info)
  })
  
  observeEvent(input$predict, {
    selected_model_type <- input$model_type
    
    # Create input data frame for prediction
    patient_data <- data.frame(
      FunctionalAssessment = input$FunctionalAssessment,
      ADL = input$ADL,
      MemoryComplaints = factor(input$MemoryComplaints, levels = c("0", "1")),
      BehavioralProblems = factor(input$BehavioralProblems, levels = c("0", "1")),
      MMSE = input$MMSE,
      CholesterolHDL = input$CholesterolHDL,
      SleepQuality = input$SleepQuality
    )
    
    # Make prediction based on the selected model
    prediction_text <- ""
    if (selected_model_type == "Logistic Regression (GLM)") {
      pred_prob <- predict(glm_model, newdata = patient_data, type = "response")
      optimal_threshold_glm <- 0.5
      prediction <- ifelse(pred_prob > optimal_threshold_glm, "Alzheimer's Disease Likely", "No Alzheimer's Disease Likely")
      prediction_text <- paste("Model:", selected_model_type, "\nPredicted Probability (AD):", round(pred_prob, 3), "\nPrediction:", prediction)
    } else if (selected_model_type == "Gradient Boosting Machine (GBM)") {
      pred_prob <- predict(gbm_model, newdata = patient_data, type = "response", n.trees = gbm_model$bestTune$n.trees)
      optimal_threshold_gboost <- 0.3567714
      prediction <- ifelse(pred_prob > optimal_threshold_gboost, "Alzheimer's Disease Likely", "No Alzheimer's Disease Likely")
      prediction_text <- paste("Model:", selected_model_type, "\nPredicted Probability (AD):", round(pred_prob, 3), "\nPrediction:", prediction)
    }
    
    output$prediction <- renderText({
      prediction_text
    })
  })
  
  # GBM Feature Importance Plot
  output$gbm_importance_plot <- renderPlot({
    if (input$model_type == "Gradient Boosting Machine (GBM)") {
      gbm_importance <- summary(gbm_model, plotit = FALSE)
      importance_df <- data.frame(
        Variable = gbm_importance$var,
        Importance = gbm_importance$rel.inf
      )
      ggplot(importance_df, aes(x = reorder(Variable, Importance), y = Importance)) +
        geom_bar(stat = "identity", fill = "steelblue") +
        coord_flip() +
        labs(title = "GBM Predictor Importance", x = "Predictor", y = "Relative Influence")
    } else {
      return(NULL) # Don't show plot if GLM is selected
    }
  })
}

# Run the application
shinyApp(ui = ui, server = server)
