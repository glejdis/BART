library(caret)            # bagging
library(dplyr)            # data wrangling
library(rpart)            # performing regression trees
library(rpart.plot)       # plotting regression trees
library(rattle)
library (gmodels)         # to compute CrossTable
library(Metrics)          # to compute RMSE

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


#' Decision tree with caret
#' 
#' Regression tree
#' 
#' Grow tree
#' Fit the model on the training set
fit <- rpart(formula=y_train~., data=x_train, method="anova")
summary(fit)

printcp(fit) # display the cp table
plotcp(fit) # visualize cross-validation results
summary(fit) # detailed summary of splits

fit$cptable

# find best value of cp
min_cp = fit$cptable[which.min(fit$cptable[,"xerror"]),"CP"]
min_cp
## 0.04269847

par(mfrow=c(1,2)) # two plots on one page
#' create additional plots
rsq.rpart(fit) # plot approximate R-squared and relative error for different splits

#' do prediction on train set
pred_fit <- predict(fit, train)
actual <- y_train
plot(pred_fit, actual, main = "Prediction on the training set - rpart" )
abline(0,1)

# RMSE on train data
rmse_fit <- rmse(y_train, pred_fit)
# 44.70872

#' R Square on train data
(cor(train$y,pred_fit))^2
# 66% 

#' prediction on test set
pred_fit_test <- predict(fit, test)
actual <- y_test
plot(pred_fit_test, actual, main = "Prediction on the test set - rpart")
abline(0,1)
# RMSE on test set
rmse_fit_test <- rmse(y_test, pred_fit_test)
# 65.78122

#' R Square on test data
(cor(test$y,pred_fit_test))^2
# 33%

#' Prune the tree using the best value of cp
pfit <- prune(fit, cp=fit$cptable[which.min(fit$cptable[,"xerror"]),"CP"])
#' Prediction on the pruned tree
Prediction_rpart<-predict(pfit,test)

# RMSE on test set
rmse_prune_test <- rmse(y_test, Prediction_rpart)
# 68.02499

#' R Square on test data
(cor(y_test,Prediction_rpart))^2
# 0.253725

set.seed(123)
#' Perform 10 fold cross validation 
#' 
#' Set up caret to perform 10-fold cross validation repeated 10 times
caret.control <- trainControl(method = "repeatedcv",
                              number = 10,
                              repeats = 10)

#' The tuneLength parameter tells the algorithm to try different default values for the main parameter cp
#' we take tuneLength = 100
rpart.cv <- train(x_train,y_train,
                  method = "rpart",
                  trControl = caret.control, tuneLength = 100)

rpart.cv 
# RMSE was used to select the optimal model using the smallest value.
# The final value used for the model was cp = 0.03304189
# with RMSE=62.83663, Rsquared=0.3580908, MAE=50.09027

#' Plot the model after performin 10-fold cross validation
plot(rpart.cv, main = " rpart - cross validation tuning parameters")

#' Pull out the the trained model using the best parameters on all the data
rpart.best <- rpart.cv$finalModel
#' Check the predictions of the model
Bestprediction <- predict(rpart.best,test)

# RMSE on test set
rmse_best_fit_test <- rmse(y_test, Bestprediction)
#  63.89937

#' R Square on test data
(cor(y_test,Bestprediction))^2

#'
#' Perform further tuning

hyper_grid <- expand.grid(
  minsplit = seq(5, 20, 1),
  maxdepth = seq(1, 15, 1)
)
# total number of combinations
nrow(hyper_grid)

models <- list()

for (i in 1:nrow(hyper_grid)) {
  
  # get minsplit, maxdepth values at row i
  minsplit <- hyper_grid$minsplit[i]
  maxdepth <- hyper_grid$maxdepth[i]
  
  # train a model and store in the list
  models[[i]] <- rpart(
    formula=y_train~., 
    data=x_train, 
    method="anova",
    control = list(minsplit = minsplit, maxdepth = maxdepth)
  )
}

# function to get optimal cp
get_cp <- function(x) {
  min    <- which.min(x$cptable[, "xerror"])
  cp <- x$cptable[min, "CP"] 
}

# function to get minimum error
get_min_error <- function(x) {
  min    <- which.min(x$cptable[, "xerror"])
  xerror <- x$cptable[min, "xerror"] 
}

hyper_grid %>%
  mutate(
    cp    = purrr::map_dbl(models, get_cp),
    error = purrr::map_dbl(models, get_min_error)
  ) %>%
  arrange(error) %>%
  top_n(-5, wt = error)

##   minsplit maxdepth         cp     error
##  1       13        3 0.02258496 0.6084414
##  2        8       13 0.02258496 0.6191598
##  3        8        3 0.02258496 0.6247567
##  4        5       14 0.02258496 0.6250213
##  5       10        8 0.03965648 0.6256289

#' we apply such results to fit our tree
optimal_tree <- rpart(
  formula=y_train~., 
  data=x_train,
  method  = "anova",
  control = list(minsplit = 13, maxdepth = 3, cp = 0.023)
)

Prediction_optimal<-predict(optimal_tree,test)

# RMSE on test set
rmse_optimal_test <- rmse(y_test, Prediction_optimal)
# 63.89937

#' R Square on test data
(cor(y_test,Prediction_optimal))^2
# 0.3271713

#' Performance evaluation
#' 
#' Examine the results of CART 
#'
#' Plot the results of first built tree: fit
fancyRpartPlot(fit, caption = 'rpart tree', cex=0.7, main = "The first Regression tree using rpart")

#' plot tree - different visualization
plot(fit, uniform=TRUE,
     main="Regression Tree for Diabetes ")
text(fit, use.n=TRUE, all=TRUE, cex=.8, digits=3)
#' Display the cp table
printcp(fit) 
#' Plot cross-validation results
plotcp(fit)
#' Plot approximate R-squared and relative error for different splits 
rsq.rpart(fit)
#' Print results
print(fit)
#' A detailed summary of splits
summary(fit) 

round(fit$cptable[which.min(fit$cptable[,"xerror"]),"CP"], 4)

gc()

#' Plot the results of the pruned tree
rpart.plot(pfit, uniform=TRUE,
           main="Pruned Regression Tree for Diabetes")

#' Plot the results of the best tree model
fancyRpartPlot(rpart.best, main = "The final Regression tree using rpart", cex=0.8)


#' Report variable importance
rpart.imp <- sort(rpart.best$variable.importance,decreasing=TRUE)

#' Plot variable importance in decreasing order
barplot(rpart.imp,las = 2, main="Variable Importance - CART", col='#005534', cex.names=0.8)


