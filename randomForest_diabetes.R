library(rsample)      # data splitting 
library(randomForest) # basic implementation
library(ranger)       # a faster implementation of randomForest
library(caret)        # an aggregator package for performing many machine learning models

#' load data
db = read.csv("http://www.rob-mcculloch.org/data/diabetes.csv")


#' 75% of the sample size
smp_size <- floor(0.75 * nrow(db))

#' set the seed to make your partition reproducible
set.seed(123)

train_ind <- sample(seq_len(nrow(db)), size = smp_size)

train <- db[train_ind, ]
test <- db[-train_ind, ]

x_train <- train[,-c(1)]
y_train <- train[,1]

x_test <- test[,-c(1)]
y_test <- test[,1]

#' Random forest
#' Build the model
#'
set.seed(123)
model1 <- randomForest(formula = y_train ~ ., data = x_train, importance = TRUE)
model1
plot(model1, main = "Plot of the first model")

#' Make predictions
prediction_rf_1 <-predict(model1, test)

# RMSE on test set
rmse_model1 <- rmse(y_test, prediction_rf_1)
## 56.80204

#' R Square on test data
(cor(y_test,prediction_rf_1))^2
# 0.4653887

# number of trees with lowest MSE
which.min(m1$mse)
## 479

# MSE of this optimal random forest
sqrt(m1$mse[which.min(m1$mse)])
## 57.71288

set.seed(123)
#' build the model
rf_oob_comp <- randomForest(formula = y_train ~ ., data = x_train, xtest = x_test,  ytest = y_test)

# extract OOB & validation errors
oob <- sqrt(rf_oob_comp$mse)
validation <- sqrt(rf_oob_comp$test$mse)

# compare error rates
tibble::tibble(
  `Out of Bag Error` = oob,
  `Test error` = validation,
  ntrees = 1:rf_oob_comp$ntree
) %>%
  gather(Metric, RMSE, -ntrees) %>%
  ggplot(aes(ntrees, RMSE, color = Metric)) +
  geom_line() +

  xlab("Number of trees") +
  labs(title = "OOB error vs. test set error")

#' Predicting on train set
predTrain1 <- predict(model1, train)

#' Predicting on test set
predTest <- predict(model1, data = y_test)

#' Perform 10-fold cross validation
#' Define the control
trControl <- trainControl(method = "cv",
                          number = 10,
                          search = "grid")

#' Run the model
rf_default <- train(x_train, y_train,
                    method = "rf",
                    metric = "RMSE",
                    trControl = trControl)
#' Print the results
print(rf_default)

##   mtry  RMSE      Rsquared   MAE     
##    2    60.55566  0.4419413  51.46033
##   33    56.86058  0.4631261  46.87749
##   64    57.19766  0.4572249  46.96175

##  RMSE was used to select the optimal model using the smallest value.
##  The final value used for the model was mtry = 33.



#' Search best mtry
tuneGrid <- expand.grid(.mtry = c(1: 50))
rf_mtry <- train(x_train, y_train,
                 method = "rf",
                 metric = "RMSE",
                 tuneGrid = tuneGrid,
                 trControl = trControl,
                 importance = TRUE,
                 nodesize = 5,
                 ntree = 500)
print(rf_mtry)


##  RMSE was used to select the optimal model using the smallest value.
##  The final value used for the model was mtry = 10
## with RMSE = 57.12804  and R-sqaured = 0.4631023 

plot(rf_mtry, main="Plot of RMSE by number of variables used in model for Random Forest")

#' The best value of mtry is stored in:
best_mtry <- rf_mtry$bestTune$mtry
## 10
min(rf_mtry$results$RMSE)
## 57.12804


#' Search the best maxnodes
store_maxnode <- list()
tuneGrid <- expand.grid(.mtry = best_mtry)
for (maxnodes in c(5: 15)) {
  set.seed(1234)
  rf_maxnode <- train(x_train, y_train,
                      method = "rf",
                      metric = "RMSE",
                      tuneGrid = tuneGrid,
                      trControl = trControl,
                      importance = TRUE,
                      nodesize = 5,
                      maxnodes = maxnodes,
                      ntree = 500)
  current_iteration <- toString(maxnodes)
  store_maxnode[[current_iteration]] <- rf_maxnode
}
results_node <- resamples(store_maxnode)
summary(results_node)

store_maxnode <- list()
tuneGrid <- expand.grid(.mtry = best_mtry)
for (maxnodes in c(20:30)) {
  set.seed(1234)
  rf_maxnode <- train(x_train, y_train,
                      method = "rf",
                      metric = "RMSE",
                      tuneGrid = tuneGrid,
                      trControl = trControl,
                      importance = TRUE,
                      nodesize = 5,
                      maxnodes = maxnodes,
                      ntree = 500)
  current_iteration <- toString(maxnodes)
  store_maxnode[[current_iteration]] <- rf_maxnode
}
results_node <- resamples(store_maxnode)
summary(results_node)

store_maxnode <- list()
tuneGrid <- expand.grid(.mtry = best_mtry)
for (maxnodes in c(35:45)) {
  set.seed(1234)
  rf_maxnode <- train(x_train, y_train,
                      method = "rf",
                      metric = "RMSE",
                      tuneGrid = tuneGrid,
                      trControl = trControl,
                      importance = TRUE,
                      nodesize = 5,
                      maxnodes = maxnodes,
                      ntree = 500)
  current_iteration <- toString(maxnodes)
  store_maxnode[[current_iteration]] <- rf_maxnode
}
results_node <- resamples(store_maxnode)
summary(results_node)

## The lowest RMSE score is obtained with a value of maxnode equals to 11



#' Search the best ntrees
store_maxtrees <- list()
for (ntree in c(10, 50, 100, 250, 300, 350, 400, 450, 500, 550, 600, 800, 1000)) {
  set.seed(5678)
  rf_maxtrees <- train(x_train, y_train,
                       method = "rf",
                       metric = "RMSE",
                       tuneGrid = tuneGrid,
                       trControl = trControl,
                       importance = TRUE,
                       nodesize = 5,
                       maxnodes = 24,
                       ntree = ntree)
  key <- toString(ntree)
  store_maxtrees[[key]] <- rf_maxtrees
}
results_tree <- resamples(store_maxtrees)
summary(results_tree)
#' Best ntree = 250

set.seed(123)
fit_rf <- randomForest(x_train, y_train,
                       method = "rf",
                       metric = "RMSE",
                       tuneGrid = tuneGrid,
                       trControl = trControl,
                       importance = TRUE,
                       nodesize = 5,
                       ntree = 50,
                       maxnodes = 11)
#' Make predictions
prediction_rf <-predict(fit_rf, test)

# RMSE on test set
rmse_best_fit_test <- rmse(y_test, prediction_rf)
rmse_best_fit_test
## 56.84108

#' R Square on test data
(cor(y_test,prediction_rf))^2
# 0.4689909

#' Plot the result from the search for the best mtry
plot(rf_mtry)

#' Plot variable importance
varImpPlot(fit_rf, main="Variable Importance - Random Forest")

