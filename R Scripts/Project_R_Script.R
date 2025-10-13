#Author: Group 15
#UBIT #: 50546666, 50547177, 50527922
#Project R Script

#Load the necessary libraries
library(ggplot2)
library(viridisLite)
library(purrr)
library(corrplot)
library(car)
library(caret)
library(randomForest)
library(xgboost)
library(dplyr)
library(ipred)
library(catboost)
library(rpart)
library(tidyverse)
library(datasets)
library(PerformanceAnalytics)

#Set working directory
setwd("C://Users//smart//OneDrive//Desktop//sid")

#Read the input dataset
df <- read.csv("myData.csv")
df1 = read.csv("myData.csv")

## EDA Start

na.count<- sum(is.na(df1))
na.count

df1[is.na(df1)]
df1 <- na.omit(df1)
sum(is.na(df1))

map_int(df1, function(x) sum(x=="[]"))

df12 <- select(df1, -c(playlist_url, track_id, artist_id, artist_genres))
View(df12)
head(df12)
str(df12)

corrln <- cor(df12[, c(7:19)])
corrln

#Draw the correlation plot
corrplot(corrln, outline = TRUE, addgrid.col = "lightblue", type="upper", method = "number")
mtext("Music Features Correlation Plot", at=5, line=-0.5, cex=1.5)


# Set up a multi-panel layout with 3 rows and 4 columns
par(mfrow=c(3, 4))
# Plot histograms for each column
hist(df12$danceability, main="Danceability", col="skyblue", border="black", xlab="Values", ylab="Frequency")
hist(df12$energy, main="Energy", col="skyblue", border="black", xlab="Values", ylab="Frequency")
hist(df12$key, main="Key", col="skyblue", border="black", xlab="Values", ylab="Frequency")
hist(df12$loudness, main="Loudness", col="skyblue", border="black", xlab="Values", ylab="Frequency")
hist(df12$mode, main="Mode", col="skyblue", border="black", xlab="Values", ylab="Frequency")
hist(df12$speechiness, main="Speechiness", col="skyblue", border="black", xlab="Values", ylab="Frequency")
hist(df12$acousticness, main="Acousticness", col="skyblue", border="black", xlab="Values", ylab="Frequency")
hist(df12$instrumentalness, main="Instrumentalness", col="skyblue", border="black", xlab="Values", ylab="Frequency")
hist(df12$liveness, main="Liveness", col="skyblue", border="black", xlab="Values", ylab="Frequency")
hist(df12$valence, main="Valence", col="skyblue", border="black", xlab="Values", ylab="Frequency")
hist(df12$tempo, main="Tempo", col="skyblue", border="black", xlab="Values", ylab="Frequency")
hist(df12$time_signature, main="Time Signature", col="skyblue", border="black", xlab="Values", ylab="Frequency")
str(df12)
# Set up a multi-panel layout with 3 rows and 4 columns
#par(mfrow=c(3, 4))
# Create boxplots for each column
boxplot(df12$danceability, main="Danceability", col="skyblue", border="black", xlab="Values", ylab="Frequency")
boxplot(df12$energy, main="Energy", col="skyblue", border="black", xlab="Values", ylab="Frequency")
boxplot(df12$key, main="Key", col="skyblue", border="black", xlab="Values", ylab="Frequency")
boxplot(df12$loudness, main="Loudness", col="skyblue", border="black", xlab="Values", ylab="Frequency")
boxplot(df12$mode, main="Mode", col="skyblue", border="black", xlab="Values", ylab="Frequency")
boxplot(df12$speechiness, main="Speechiness", col="skyblue", border="black", xlab="Values", ylab="Frequency")
boxplot(df12$acousticness, main="Acousticness", col="skyblue", border="black", xlab="Values", ylab="Frequency")
boxplot(df12$instrumentalness, main="Instrumentalness", col="skyblue", border="black", xlab="Values", ylab="Frequency")
boxplot(df12$liveness, main="Liveness", col="skyblue", border="black", xlab="Values", ylab="Frequency")
boxplot(df12$valence, main="Valence", col="skyblue", border="black", xlab="Values", ylab="Frequency")
boxplot(df12$tempo, main="Tempo", col="skyblue", border="black", xlab="Values", ylab="Frequency")
boxplot(df12$time_signature, main="Time Signature", col="skyblue", border="black", xlab="Values", ylab="Frequency")

dev.off()
par(mfrow=c(2,2))
plot(df12$valence, df12$energy,
     xlab = "Valence",
     ylab = "Energy",
     main = "Scatter Plot of Valence vs. Energy",
     pch = 19, # Point shape
     col = "blue" # Point color
)
abline(lm(df12$energy ~ df12$valence), col = "red", lwd = 2)
plot(df12$loudness, df12$energy,
     xlab = "Loudness",
     ylab = "Energy",
     main = "Scatter Plot of Loudness vs. Energy",
     pch = 19, # Point shape
     col = "red" # Point color
)
abline(lm(df12$energy ~ df12$loudness), col = "blue", lwd = 2)
plot(df12$valence, df12$danceability,
     xlab = "Valence",
     ylab = "Danceability",
     main = "Scatter Plot of Valence vs. Danceability",
     pch = 19, # Point shape
     col = "pink" # Point color
)
abline(lm(df12$valence ~ df12$danceability), col = "blue", lwd = 2)
plot(df12$acousticness, df12$energy,
     xlab = "Acousticness",
     ylab = "Energy",
     main = "Scatter Plot of Acousticness vs. Energy",
     pch = 19, # Point shape
     col = "green" # Point color
)
abline(lm(df12$acousticness ~ df12$energy), col = "red", lwd = 2)

chart.Correlation(df12[, 7:19], histogram = TRUE, method = "pearson")

pairs(~ danceability + energy + key + loudness + mode + speechiness + acousticness + instrumentalness + liveness +
        valence + tempo + time_signature,
      data = df12,
      col = c("black","orange","purple"),
      upper.panel = NULL,    # Correlation panel
      lower.panel = panel.smooth,
      #diag.panel = panel.hist,
      pch = 18)

summary(df12)

df13 <- select(df12, -c(year, track_name, track_popularity, album, artist_name, artist_popularity))

scaled_data <- scale(df13)

pca_result <- prcomp(scaled_data)

variance_explained <- pca_result$sdev^2
variance_explained_ratio <- variance_explained / sum(variance_explained)

variance_explained

n <- 10
selected_features <- df13[, pca_result$rotation[, 1:n]]

selected_features

reduced_data <- pca_result$x
reduced_data

## EDA Complete

#Get summary of data
summary(df)

df_numeric <- subset(df, select = -c(playlist_url, year, track_id, track_popularity, album, artist_id
                                     , artist_name, artist_popularity, duration_ms, mode, time_signature, key, track_name))

summary(df_numeric)
str(df_numeric)

unique(df_numeric$artist_genres)

ggplot(df_numeric, aes(x = artist_genres, fill = artist_genres)) +
  geom_bar() +
  labs(x = 'Mean Genre Encoded', y = 'Count', title = 'Class Label Count') + 
  scale_fill_viridis_d() + 
  theme_classic()

sum(is.na(df_numeric))

df_numeric <- na.omit(df_numeric)

df_numeric$upd <- as.numeric(factor(df_numeric$artist_genres))

str(df_numeric)

df_numeric$upd <- ifelse(df_numeric$upd %in% c(11, 15, 6), df_numeric$upd, 1) 

unique(df_numeric$upd)

df_numeric$upd <- as.factor(df_numeric$upd)

ggplot(df_numeric, aes(x = upd, y = after_stat(count), fill = upd)) +
  geom_bar() +
  geom_text(stat = "Count", aes(label = ifelse(upd == "1", "Others", artist_genres)), vjust = -0.5, color = "black") +
  labs(x = 'Encoded Genres', y = 'Count', title = 'Encoded Genre Class') +
  scale_fill_viridis_d() + 
  theme_classic()

selected_rows <- df_numeric

selected_rows$upd <- as.numeric(selected_rows$upd)

#1 - Others
#2 - Hip-Hop
#3 - Pop
#4 - Rock

summary(selected_rows)
str(selected_rows)

correlation_features <- cor(selected_rows[, c(2:10)])

corrplot(correlation_features, method = "number", title = "Correlation Plot", outline = TRUE, addgrid.col = "lightgrey", order="hclust")

corr_model <- lm(upd ~ danceability + energy + loudness + speechiness + acousticness
                 + instrumentalness + liveness + valence + tempo, data = selected_rows)
vif_values <- vif(corr_model)
vif_values

selected_rows <- subset(selected_rows, select = -c(artist_genres, loudness, tempo, liveness))
selected_rows[, c("danceability", "energy", "speechiness", "acousticness", "instrumentalness", "valence")] <- scale(selected_rows[, c("danceability", "energy", "speechiness", "acousticness", "instrumentalness", "valence")])

#Filter 'upd' values to [1, 5, 6]
selected_rows <- subset(selected_rows, upd %in% c(2, 3, 4))

#Use the top 5 correlated features with 'upd'
top_features <- c('danceability', 'speechiness', 'energy', 'acousticness', 'instrumentalness')

selected_rows$upd <- as.factor(selected_rows$upd)

##Implementing the randomForestClassifier

# Split the data into training and testing sets (70-30 split)
set.seed(123)

train_indices <- sample(1:nrow(selected_rows), 0.75 * nrow(selected_rows))
train_data <- selected_rows[train_indices, ]
test_data <- selected_rows[-train_indices, ]
X_train <- train_data[top_features]
X_test <- test_data[top_features]
y_train <- train_data["upd"]
y_test <- test_data["upd"]

#View(train_data)
#View(test_data)

#Create a Random Forest classifier with reduced tree depth
rf_model <- randomForest(upd ~ ., data = train_data, ntree = 30,
                         mtry = length(top_features), maxdepth = 5, nodesize = 1, importance = TRUE, seed = 42)

#Print the model summary
print(rf_model)


#Perform cross-validation on the entire dataset
cv_results <- train(upd ~ ., data = selected_rows, method = "rf",
                    trControl = trainControl(method = "cv", number = 5),
                    tuneGrid = data.frame(mtry = length(top_features)))

#Access cross-validation results
cv_scores <- cv_results$results$Accuracy  # Change 'Accuracy' to the appropriate metric
cv_scores

#Predict on the test set
y_pred <- predict(rf_model, newdata = test_data, type = "response")
y_pred

#Convert predicted and true labels to factors
y_pred <- as.factor(y_pred)
y_actual <- as.factor(test_data$upd)

#Create a confusion matrix
conf_matrix_rf <- table(Predicted = y_pred, Actual = y_actual)
print(conf_matrix_rf)
accuracy_rf <- sum(diag(conf_matrix_rf)) / sum(conf_matrix_rf)
print(paste("Accuracy:", accuracy_rf*100))

##Implementing XGBoosting

y_train_xgb <- y_train %>%
  mutate(upd = case_when(
    upd == 2 ~ 0,
    upd == 3 ~ 1,
    upd == 4 ~ 2,
    TRUE ~ as.numeric(as.character(upd))
  ))

y_test_xgb <- y_test %>%
  mutate(upd = case_when(
    upd == 2 ~ 0,
    upd == 3 ~ 1,
    upd == 4 ~ 2,
    TRUE ~ as.numeric(as.character(upd))
  ))

y_train_xgb <- y_train_xgb$upd
y_test_xgb <- y_test_xgb$upd

xgb_num_class <- length(unique(y_train_xgb))

length(y_train_xgb)
nrow(X_train)
str(y_train_xgb)
str(y_test_xgb)

xgb_dtrain <- xgb.DMatrix(as.matrix(X_train), label = y_train_xgb)

xgb_params <- list(
  objective = "multi:softmax",
  max_depth = 5,
  alpha = 0.5,
  lambda = 0.5,
  gamma = 1,
  num_class = xgb_num_class,
  eval_metric = "mlogloss"
)

xgb_model <- xgboost(params = xgb_params, data = xgb_dtrain, nrounds = 10)

xgb_dtest <- xgb.DMatrix(as.matrix(X_test))

y_pred_xgb <- predict(xgb_model, xgb_dtest)

y_test_xgb <- as.factor(y_test_xgb)

y_pred_xgb <- factor(y_pred_xgb, levels = levels(y_test_xgb))

confusion_matrix_xgb <- confusionMatrix(y_pred_xgb, y_test_xgb)
accuracy_xgb <- confusion_matrix_xgb$overall["Accuracy"]
accuracy_xgb <- as.numeric(accuracy_xgb)
accuracy_xgb

## Implementing AdaBoost model

#Convert labels to start from 0
y_train_adaboost <- y_train %>%
  mutate(upd = case_when(
    upd == 2 ~ 1,
    upd == 3 ~ 2,
    upd == 4 ~ 3,
    TRUE ~ as.numeric(as.character(upd))
  )) %>%
  pull(upd)

y_test_adaboost <- y_test %>%
  mutate(upd = case_when(
    upd == 2 ~ 1,
    upd == 3 ~ 2,
    upd == 4 ~ 3,
    TRUE ~ as.numeric(as.character(upd))
  )) %>%
  pull(upd)

train_data_ada <- train_data
test_data_ada <- test_data

train_data_ada$upd <- as.numeric(as.character(train_data_ada$upd))
train_data_ada$upd <- ifelse(train_data_ada$upd == 2, 1, ifelse(train_data_ada$upd == 3, 2, ifelse(train_data_ada$upd == 4, 3, train_data_ada$upd)))
train_data_ada$upd <- as.factor(train_data_ada$upd)

set.seed(42)
adaboost_model <- bagging(upd ~ danceability + speechiness + energy + acousticness + instrumentalness, data = train_data_ada, iter = 100, nbag = 50)

y_pred_adaboost <- predict(adaboost_model, newdata = X_test)

y_pred_adaboost <- as.factor(y_pred_adaboost)
y_test_ada <- y_test

class(y_pred_adaboost)
class(y_test_ada)

confusion_matrix_adaboost <- table(y_pred_adaboost, y_test_ada$upd)
accuracy_adaboost <- sum(diag(confusion_matrix_adaboost)) / sum(confusion_matrix_adaboost)
accuracy_adaboost

## Implementing Decision Tree Classifier

#Create a decision tree model
decision_tree_model <- rpart(upd ~ ., data = train_data, method = "class")

#Make predictions on the test set
y_pred_decision_tree <- predict(decision_tree_model, newdata = test_data, type = "class")

#Convert predicted labels to factor
y_pred_decision_tree <- as.factor(y_pred_decision_tree)

#Evaluate accuracy
confusion_matrix_decision_tree <- table(y_pred_decision_tree, y_test$upd)
accuracy_decision_tree <- sum(diag(confusion_matrix_decision_tree)) / sum(confusion_matrix_decision_tree)
accuracy_decision_tree

## Implementing CatBoost model

#Convert 'upd' column to numeric type
train_data$upd <- as.numeric(as.factor(train_data$upd))
test_data$upd <- as.numeric(as.factor(test_data$upd))

#Create CatBoost pools with the updated data
train_pool <- catboost.load_pool(data = as.matrix(train_data[top_features]), label = train_data$upd)
test_pool <- catboost.load_pool(data = as.matrix(test_data[top_features]), label = test_data$upd)

#Define Parameters
params <- list(
  iterations = 100,
  depth = 5,
  learning_rate = 0.1,
  loss_function = 'MultiClass',
  verbose = 10
)

# Step 5: Train the Model
catboost_model <- catboost.train(train_pool, params = params)

# Make predictions on the test set
y_pred_catboost_probs <- catboost.predict(catboost_model, test_pool)

# Convert probabilities to class labels
# Assuming that class labels are 0, 1, 2, ..., n-1 for a n-class problem
y_pred_catboost <- apply(y_pred_catboost_probs, 1, which.max) - 1

# Evaluate the Model
confusion_matrix_catboost <- table(Predicted = y_pred_catboost, Actual = y_test$upd)
accuracy_catboost <- sum(diag(confusion_matrix_catboost)) / sum(confusion_matrix_catboost)

colnames(confusion_matrix_catboost) <- c("Predicted:2", "Predicted:3", "Predicted:4")
rownames(confusion_matrix_catboost) <- c("Actual:2", "Actual:3", "Actual:4")

#Calculate precision, recall, and F1 score
precision <- recall <- f1_score <- numeric(3)

for (i in 1:3) {
  tp <- confusion_matrix_catboost[i, i]
  fp <- sum(confusion_matrix_catboost[-i, i])
  fn <- sum(confusion_matrix_catboost[i, -i])
  precision[i] <- tp / (tp + fp)
  recall[i] <- tp / (tp + fn)
  f1_score[i] <- 2 * (precision[i] * recall[i]) / (precision[i] + recall[i])
}

#Create a dataframe for plotting
metrics_data <- data.frame(
  Class = factor(rep(2:4, each = 3)),
  Metric = rep(c("Precision", "Recall", "F1 Score"), times = 3),
  Value = c(precision, recall, f1_score)
)

#Plot the results of CatBoosting model
ggplot(metrics_data, aes(x = Class, y = Value, fill = Metric)) +
  geom_bar(stat = "identity", position = position_dodge()) +
  labs(title = "CatBoost Model Performance Metrics",
       x = "Class",
       y = "Value") +
  scale_fill_brewer(palette = "Set1") +
  theme_classic()

#Print the accuracies
print(accuracy_decision_tree)
print(accuracy_adaboost)
print(accuracy_xgb)
print(accuracy_catboost)
print(accuracy_rf)