#set library, allocated RAM ( 2500 M ) and number 
#of computational cores
options(java.parameters = "-Xmx2500m")
library("bartMachine")
set_bart_machine_num_cores(2)

#get automobile data included within bartMachine package
data("automobile", package = "bartMachine")

#omit NA entries. Y = log price, X = 21 car features
automobile <- na.omit(automobile)
y <- automobile$log_price
X <- automobile; X$log_price <- NULL

#build the bartMachine from X and y
bart_machine <- bartMachine(X, y)
#look at the statistics of the data and the bartMachine
bart_machine

#perofrm k fold validation for unseen data
k_fold_cv(X, y, k_folds = 10)
rmse_by_num_trees(bart_machine, num_replicates = 20)

#find the best hyperparameters and then save the 
#bartMachine initialized with the "best" hyperparameters
bart_machine_cv <- bartMachineCV(X, y)
k_fold_cv(X, y, k_folds = 10, k = 2, nu = 3, q = 0.9, num_trees = 200)

#check error assumptions (QQ plot of normality of residual)
# and MCMC convetgences
check_bart_error_assumptions(bart_machine_cv)
plot_convergence_diagnostics(bart_machine_cv)

#calculate credible intervals for the 100th entry
round(calc_credible_intervals(bart_machine_cv, X[100, ],
                              ci_conf = 0.95), 2)
#plot y vs yhat with credible intervals & prediction int
plot_y_vs_yhat(bart_machine_cv, credible_intervals = TRUE)
plot_y_vs_yhat(bart_machine_cv, prediction_intervals = TRUE)

#investigate variable importance, plot PD plots of selected variates
investigate_var_importance(bart_machine_cv, num_replicates_for_avg = 20)
pd_plot(bart_machine_cv, j = "horsepower")
pd_plot(bart_machine_cv, j = "stroke")

#select the most important variables
vs <- var_selection_by_permute(bart_machine_cv,
        bottom_margin = 5, num_permute_samples = 10)
vs$important_vars_local_names
vs$important_vars_global_max_names
vs$important_vars_global_se_names

#serialize and save
bart_machine <- bartMachine(X, y, serialize = TRUE)
save.image("bart_demo.RData")
q("no")