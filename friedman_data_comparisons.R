library(bartMachine) #bartMachine library for BART model
library(rpart) #regression tree library
library(Metrics) #for calculating RMSE
library(randomForest) #random forest library

# function to generate friedman data 
#input n = number of observations,
#p = number of covariates
#sigma = variance of the normal noise distribution
gen_friedman_data = function(n, p, sigma){
  # if p < 5, exit the function and print out the following
  if (p < 5){stop("p must be greater than or equal to 5")}
  
  # generate X from a random uniform distribution
  # n observations and p covariates
  X <- matrix(runif(n * p), nrow = n, ncol = p)
  
  # calculate y response based on the Friedman equation
  y <- 10 * sin(pi * X[, 1] * X[, 2]) + 20 * (X[, 3] - .5)^2 +
  10 * X[, 4] + 5 * X[, 5] + rnorm(n, 0, sigma)
  
  #return data frame containing X and y response
  data.frame(y, X)
}
# take 100 covariates, of which the first 5 covariates are 
# the only important covariates
p = 100
sigma = 1 

#take 500 training data and 500 test data
ntrain = 500
ntest = 500

#set seed for reproducibility
set.seed(12456)

#generate training friedman data
fr_data <- gen_friedman_data(ntrain, p, sigma)
y_train <- fr_data$y
X_train <- fr_data[, 2:101]

#generate test friedman data
fr_data <- gen_friedman_data(ntest, p, sigma)
y_test <- fr_data$y
X_test <- fr_data[, 2:101]

#train bartmachine model with training dataset
bm_friedman <- bartMachine(X_train, y_train)
#get test set prediction, RMSE and R^2
pred_bart <- bart_predict_for_test_data(bm_friedman, X_test, y_test)$y_hat
rmse_bart <- bart_predict_for_test_data(bm_friedman, X_test, y_test)$rmse
# 1.619262
(cor(y_test,pred_bart))^2
# 0.882

#train regression tree with training dataset
fit <- rpart(formula=y_train~., data=X_train, method="anova")
#get test set prediction, RMSE and R^2
pred_fit_test <- predict(fit, X_test)
rmse_fit_test <- rmse(y_test, pred_fit_test)
# 2.930315
(cor(y_test,pred_fit_test))^2
# 0.617

#train random forest with training dataset
model1 <- randomForest(formula = y_train ~ ., data = X_train, importance = TRUE)
#get test set prediction, RMSE and R^2
prediction_rf_1 <-predict(model1, X_test)
rmse_model1 <- rmse(y_test, prediction_rf_1)
#2.509288
(cor(y_test,prediction_rf_1))^2
#0.784
