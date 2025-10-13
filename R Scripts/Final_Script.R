setwd("/Users/siddharthhaveliwala/Documents/Fall 2023/IE 500 - PM & DA/Project/Dataset")
df <- read.csv("myData.csv")

summary(df)

df_numeric <- subset(df, select = -c(playlist_url, year, track_id, track_popularity, album, artist_id
                            , artist_name, artist_popularity, duration_ms, mode, time_signature, key, track_name))

summary(df_numeric)
str(df_numeric)
#View(df_numeric)

unique(df_numeric$artist_genres)

library(ggplot2)
#install.packages("viridisLite")
#library(viridisLite)

ggplot(df_numeric, aes(x = artist_genres, fill = artist_genres)) +
  geom_bar() +
  labs(x = 'Mean Genre Encoded', y = 'Count', title = 'Class Label Count') + 
  scale_fill_viridis_d() + 
  theme_classic()

sum(is.na(df_numeric))

df_numeric <- na.omit(df_numeric)

df_numeric$upd <- as.numeric(factor(df_numeric$artist_genres))

str(df_numeric)
#View(df_numeric)

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

library(corrplot)
corrplot(correlation_features, method = "number", title = "Correlation Plot", outline = TRUE, addgrid.col = "lightgrey", order="hclust")

library(car)
corr_model <- lm(upd ~ danceability + energy + loudness + speechiness + acousticness
                 + instrumentalness + liveness + valence + tempo, data = selected_rows)
vif_values <- vif(corr_model)
vif_values

selected_rows <- subset(selected_rows, select = -c(artist_genres, loudness, tempo, liveness))
#selected_features <- c("danceability", "energy", "speechiness", "acousticness", "instrumentalness", "valence")
selected_rows[, c("danceability", "energy", "speechiness", "acousticness", "instrumentalness", "valence")] <- scale(selected_rows[, c("danceability", "energy", "speechiness", "acousticness", "instrumentalness", "valence")])
#X <- selected_rows[selected_features]
#X <- scale(X)
#y <- selected_rows["upd"]
#View(selected_rows)

library(caret)
library(randomForest)

#set.seed(123)

#train_index <- createDataPartition(selected_rows$upd, p = 0.75, list = FALSE)
#folds <- createFolds(selected_rows$upd[train_index], k = 5, list = TRUE, returnTrain = FALSE)

#train_fold <- selected_rows[train_index[-folds[[1]]], ]
#test_fold <- selected_rows[train_index[folds[[1]]], ]

# Assuming 'selected_rows' is your dataframe
# Filter 'upd' values to [1, 5, 6]
selected_rows <- subset(selected_rows, upd %in% c(2, 3, 4))

# Use the top 5 correlated features with 'upd'
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

# Create a Random Forest classifier with reduced tree depth
rf_model <- randomForest(upd ~ ., data = train_data, ntree = 30,
                         mtry = length(top_features), maxdepth = 5, nodesize = 1, importance = TRUE, seed = 42)

# Print the model summary
print(rf_model)


# Perform cross-validation on the entire dataset
cv_results <- train(upd ~ ., data = selected_rows, method = "rf",
                    trControl = trainControl(method = "cv", number = 5),
                    tuneGrid = data.frame(mtry = length(top_features)))

# Access cross-validation results
cv_scores <- cv_results$results$Accuracy  # Change 'Accuracy' to the appropriate metric
cv_scores
# Make predictions on the test set
y_pred <- predict(rf_model, newdata = test_data, type = "response")
y_pred

# Convert predicted and true labels to factors
y_pred <- as.factor(y_pred)
y_actual <- as.factor(test_data$upd)

# Create a confusion matrix
conf_matrix_rf <- table(Predicted = y_pred, Actual = y_actual)
print(conf_matrix_rf)
accuracy_rf <- sum(diag(conf_matrix_rf)) / sum(conf_matrix_rf)
print(paste("Accuracy:", accuracy_rf*100))

##Implementing XGBoosting

#install.packages("xgboost")
library(xgboost)
library(dplyr)

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
#length(y_train)
#nrow(X_train)
#y_train_xgb <- as.factor(y_train_xgb)
#y_test_xgb <- as.factor(y_test_xgb)

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
accuracy_xgb

##adaboost model

# Install and load necessary packages
# install.packages("adabag")
#install.packages("ipred")
library(ipred)

# Convert labels to start from 0
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

##Decision Tree Classifier


library(rpart)

# Assuming train_data_ada is your training data
# Assuming upd is the target variable, and other variables are features

# Create a decision tree model
decision_tree_model <- rpart(upd ~ ., data = train_data, method = "class")

# Make predictions on the test set
y_pred_decision_tree <- predict(decision_tree_model, newdata = test_data, type = "class")

# Convert predicted labels to factor
y_pred_decision_tree <- as.factor(y_pred_decision_tree)

# Evaluate accuracy
confusion_matrix_decision_tree <- table(y_pred_decision_tree, y_test$upd)
accuracy_decision_tree <- sum(diag(confusion_matrix_decision_tree)) / sum(confusion_matrix_decision_tree)
accuracy_decision_tree

##Catboosting model


print(accuracy_decision_tree)
print(accuracy_adaboost)
print(accuracy_rf)
print(accuracy_xgb)