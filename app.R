library(shiny)
library(gbm)
library(caret)
library(pROC)
library(ggplot2)

# Load the trained models (adjust paths if needed)
glm_model <- readRDS("glm_model.rds")
gbm_model <- readRDS("gbm_model.rds");


# Function to make some nice tooltips
make_tooltip <- function(items) {
  # items: a named character vector or list
  # returns one string with each “name: value” on its own line
  entries <- sprintf("%s: %s",
                     names(items),
                     unlist(items, 
                            use.names = FALSE)
                     )
  paste(entries, collapse = "\n")
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
  sidebarLayout(
    sidebarPanel(
      selectInput("model_type", "Select Model:",
                  choices = c("Logistic Regression (GLM)", "Gradient Boosting Machine (GBM)")),
      hr(), 
      h4("Patient Data Input"),
      h5("Hower over each for more info"),
      numericInput("FunctionalAssessment", 
                   label = tags$span(
                     "FunctionalAssessment (0–10)",
                     title = paste("Add 1 for each activity where severe assistance is needed:",
                                   make_tooltip(fs_items),
                                   sep = "\n\n"
                     )
                   ), 
                   value = 0, min = 0, max = 10),
      numericInput("ADL",
                   label = tags$span(
                                      "ADL - Activities of Daily Living (0–10)",
                                      title = paste("Add 1 for each activity where assistance is needed:",
                                                    make_tooltip(adl_items),
                                                    sep = "\n\n"
                                                   )
                                    ), 
                    value = 0, min = 0, max = 10),
      tags$div(
        title = "Has the patient or caregiver noticed any memory lapses?\n 0 = No, 1 = Yes",
        selectInput(
          "MemoryComplaints",
          "Memory Complaints",
          choices = c("0", "1")
        )
      ),
      
      tags$div(
        title = "Any new or worsening behavioral symptoms (agitation, apathy, etc.)?\n 0 = No, 1 = Yes",
        selectInput(
          "BehavioralProblems",
          "Behavioral Problems",
          choices = c("0", "1")
        )
      ),
      numericInput("MMSE", 
                   label = tags$span(
                     "MSE - Mini-Mental State Exam (0–30)",
                     title = paste("Substract 1 from 30 for each failed task:",
                                   make_tooltip(mmse_items),
                                   sep = "\n\n"
                     )
                   ), 
                   value = 0, min = 0, max = 30),
      numericInput("CholesterolHDL", label   = tags$span(
                   "HDL Cholesterol (mg/dL)",
                   title = "HDL (good) cholesterol helps clear plaque from your arteries. Higher for women due to effects of estrogen on lipid metabolism - process of synthesizing, breaking down, and storing fat in cells for energy storage\n\
Aim for ≥40 mg/dL (men) or ≥50 mg/dL (women); 60+ mg/dL is optimal."
      ),
      value = 20, min = 20, max = 100),
      # We did not find any studies with index in ranges 4-10. So we decided to interpret this number as average daily sleep time from 4-10 hours.
      numericInput("SleepQuality", 
                   label = tags$span(
                     "Sleep Quality (4–10)",
                     title = "Average number of hours slept"
                   ), 
                   value = 4, min = 4, max = 10),
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
        geom_bar(stat = "identity", fill = "lightblue") +
        coord_flip() +
        labs(title = "GBM Predictor Importance", x = "Predictor", y = "Relative Influence")
    } else {
      return(NULL) # Don't show plot if GLM is selected
    }
  })
}

# Run the application
shinyApp(ui = ui, server = server)
