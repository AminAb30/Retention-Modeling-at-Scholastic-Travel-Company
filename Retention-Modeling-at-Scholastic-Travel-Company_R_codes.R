# IDS572-HW4
# Author: Amin Abbasi
# Date: 2024-11-23

# Install required packages
#install.packages("smotefamily")
#install.packages("DMwR2")
#install.packages("caret")
#install.packages("rpart")   # Decision trees
#install.packages("randomForest") # Random Forest
#install.packages("e1071") # Naive Bayes
#install.packages("adabag")  # AdaBoost

# Load libraries
library(readxl)  # For reading Excel files
library(smotefamily)
library(DMwR2)
library(caret)
library(ggplot2)
library(reshape2)
library(rpart)
library(pROC) # For ROC curve
library(randomForest)
library(e1071)
library(adabag)

# Import Dataset
data <- read_excel("Retention Modeling at Scholastic Travel Company (A)_ Student Spreadsheet.xlsx", sheet = "Data")
data$Retained.in.2012. <- as.factor(data$Retained.in.2012.)

# Data Exploration
str(data)
summary(data)

# Data Preprocessing: Remove unwanted columns
data <- data[, !names(data) %in% c("Return.Date", "Deposit.Date", "Departure.Date", "Early.RPL", 
                                   "Latest.RPL", "FirstMeeting", "LastMeeting", 
                                   "Initial.System.Date", "ID")]

# Replace empty strings or "NA" with actual NA
data <- data.frame(lapply(data, function(column) {
  column[column == "" | column == "NA"] <- NA
  return(column)
}), stringsAsFactors = FALSE)

# Handling Missing Values
calculate_mode <- function(column) {
  unique_values <- unique(column[!is.na(column)])
  mode_value <- unique_values[which.max(tabulate(match(column, unique_values)))]
  return(mode_value)
}

numeric_columns <- sapply(data, is.numeric)
categorical_columns <- sapply(data, is.character)

data[numeric_columns] <- lapply(data[numeric_columns], function(column) {
  ifelse(is.na(column), median(column, na.rm = TRUE), column)
})
data[categorical_columns] <- lapply(data[categorical_columns], function(column) {
  column[is.na(column)] <- calculate_mode(column)
  return(column)
})

# Outlier Handling: Cap at IQR thresholds
handle_outliers_iqr_cap <- function(data) {
  numeric_columns <- sapply(data, is.numeric)
  data[numeric_columns] <- lapply(data[numeric_columns], function(column) {
    Q1 <- quantile(column, 0.25, na.rm = TRUE)
    Q3 <- quantile(column, 0.75, na.rm = TRUE)
    IQR <- Q3 - Q1
    lower_bound <- Q1 - 1.5 * IQR
    upper_bound <- Q3 + 1.5 * IQR
    column[column < lower_bound] <- lower_bound
    column[column > upper_bound] <- upper_bound
    return(column)
  })
  return(data)
}
data <- handle_outliers_iqr_cap(data)

# Handling Rare Categories
threshold <- 0.005
categorical_columns <- sapply(data, is.character)
data[, categorical_columns] <- lapply(names(data)[categorical_columns], function(col_name) {
  column <- data[[col_name]]
  freq_table <- prop.table(table(column))
  rare_values <- names(freq_table[freq_table < threshold])
  column[column %in% rare_values] <- "Other"
  return(column)
})

# Check Balance
table(data$Retained.in.2012.)
prop.table(table(data$Retained.in.2012.))

# Balancing with Undersampling
downsampled_data <- downSample(x = data[, -which(names(data) == "Retained.in.2012.")],
                               y = as.factor(data$Retained.in.2012.))

# Normalization
normalize_minmax <- function(x) {
  (x - min(x, na.rm = TRUE)) / (max(x, na.rm = TRUE) - min(x, na.rm = TRUE))
}
numeric_columns <- sapply(data, is.numeric)
data[numeric_columns] <- lapply(data[numeric_columns], normalize_minmax)

# Train-Test Split
set.seed(123)
train_index <- createDataPartition(data$Retained.in.2012., p = 0.8, list = FALSE)
train_data <- data[train_index, ]
test_data <- data[-train_index, ]

# Decision Tree
dt_model <- rpart(Retained.in.2012. ~ ., data = train_data, method = "class")
dt_predictions <- predict(dt_model, test_data, type = "class")
confusionMatrix(dt_predictions, test_data$Retained.in.2012.)
dt_probabilities <- predict(dt_model, test_data, type = "prob")[, 2]
dt_roc <- roc(test_data$Retained.in.2012., dt_probabilities)
plot(dt_roc, col = "blue", main = "ROC Curve for Decision Tree")
auc(dt_roc)

# Random Forest
rf_model <- randomForest(Retained.in.2012. ~ ., data = train_data)
rf_predictions <- predict(rf_model, test_data)
confusionMatrix(rf_predictions, test_data$Retained.in.2012.)
rf_probabilities <- predict(rf_model, test_data, type = "prob")[, 2]
rf_roc <- roc(test_data$Retained.in.2012., rf_probabilities)
plot(rf_roc, col = "green", main = "ROC Curve for Random Forest")
auc(rf_roc)

# Naive Bayes
nb_model <- naiveBayes(Retained.in.2012. ~ ., data = train_data)
nb_predictions <- predict(nb_model, test_data)
confusionMatrix(nb_predictions, test_data$Retained.in.2012.)
nb_probabilities <- predict(nb_model, test_data, type = "raw")[, 2]
nb_roc <- roc(test_data$Retained.in.2012., nb_probabilities)
plot(nb_roc, col = "red", main = "ROC Curve for Naive Bayes")
auc(nb_roc)

# Logistic Regression
lr_model <- glm(Retained.in.2012. ~ ., data = train_data, family = "binomial")
lr_probabilities <- predict(lr_model, test_data, type = "response")
lr_predictions <- ifelse(lr_probabilities > 0.5, 1, 0)
confusionMatrix(as.factor(lr_predictions), test_data$Retained.in.2012.)
lr_roc <- roc(test_data$Retained.in.2012., lr_probabilities)
plot(lr_roc, col = "purple", main = "ROC Curve for Logistic Regression")
auc(lr_roc)

# Compare ROC Curves
plot(dt_roc, col = "blue", main = "ROC Curves for All Models", lwd = 2)
plot(rf_roc, col = "green", add = TRUE, lwd = 2)
plot(nb_roc, col = "red", add = TRUE, lwd = 2)
plot(lr_roc, col = "purple", add = TRUE, lwd = 2)
legend("bottomright", legend = c("Decision Tree", "Random Forest", "Naive Bayes", "Logistic Regression"),
       col = c("blue", "green", "red", "purple"), lwd = 2)

# Best Model: Random Forest

