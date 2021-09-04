library("bartMachine")

#set JAVA and bartmachine parallelization parameters
options(java.parameters = "-Xmx2500m")
set_bart_machine_num_cores(4)

#read the diabetes datasets
db = read.csv("http://www.rob-mcculloch.org/data/diabetes.csv")
load("bart_machine.RData")

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

#train bart_machine from train datasets
bart_machine <- bartMachine(x_train, y_train)

#check in sample statistics
bart_machine

#plot in sample prediction statistics with credible intervals and prediction intervals
plot_y_vs_yhat(bart_machine, credible_intervals = TRUE)
plot_y_vs_yhat(bart_machine, prediction_intervals = TRUE)

#evaluate k_fold cross validation
k_fold_cv(x_train, y_train, k_folds = 10)

#plot rmse by number of trees
rmse_by_num_trees(bart_machine, num_replicates = 10)

#perform test set prediction of trained model and evaluate the RMSE
oos_perf = bart_predict_for_test_data(bart_machine, x_test, y_test)
print(oos_perf$rmse)

# Get RMSE, test predictions and calculate R^2 from prediction values
rmse_bm = oos_perf$rsme
pred_bm = oos_perf$y_hat
(cor(y_test,pred_bm))^2

#check error asumptions and convergence diagnostics of the bart model
check_bart_error_assumptions(bart_machine)

#plot MCMC convergence diagnostics
plot_convergence_diagnostics(bart_machine)


#perform hyperparameter optimization to get the best
#performing bart machine model according to k fold cv value
#note that the best performing hyperparameter values may change across
#different optimization runs
bart_machine_cv <- bartMachineCV(x_train, y_train)

#check in sample statistics
bart_machine_cv

#evaluate k_fold cross validation of the best performing hyperparameters
k_fold_cv(x_train, y_train, k_folds = 10, k = 3, nu = 3, q = 0.99, num_trees = 50)

#perform test set prediction of the trained optimized model
oos_perf = bart_predict_for_test_data(bart_machine_cv, x_test, y_test)
print(oos_perf$rmse)

# Get RMSE, test predictions and calculate R^2 from prediction values
rmse_bm = oos_perf$rsme
pred_bm = oos_perf$y_hat
(cor(y_test,pred_bm))^2

#  generate plot of covariate importance 
investigate_var_importance(bart_machine, num_replicates_for_avg = 20)

#generate plots of partial dependences for "bmi"
#and map covariates
pd_plot(bart_machine_cv, j = "bmi")
pd_plot(bart_machine_cv, j = "map")

#compare with BART from the BART library
library("BART")

#train a default bart model with the default hyperparameters
bfTrain = wbart(x_train,y_train,x_test)

#a function to calculate RMSE
rmsef = function(y,yhat) {return(sqrt(mean((y-yhat)^2)))}

#get test prediction and calculate RMSE and R^2
yBhat = bfTrain$yhat.test.mean
rmseB = rmsef(y_test,yBhat)
(cor(y_test,yBhat))^2