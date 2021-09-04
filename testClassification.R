#get Pima.te data included within bartMachine package
data("Pima.te", package = "MASS")
X <- data.frame(Pima.te[, -8])
y <- Pima.te[, 8]

#get the best hyperparmeters 
bart_machine_cv <- bartMachineCV(X, y)
bart_machine_cv

#creat bartMachine with higher prob rule class and check
bartMachine(X, y, prob_rule_class = 0.6)

#k fold cross validation and display confusion matrix
oos_stats <- k_fold_cv(X, y, k_folds = 10)
oos_stats$confusion_matrix

#just to show that you can predict either class labels or probabilities
round(predict(bart_machine_cv, X[1 : 2, ], type = "prob"), 3)
predict(bart_machine_cv, X[1:2, ], type = "class")

#calculate CI and pd plot, most of the functionality
#is similar to regression,however no PI
#since no noise assosiated with classification prediction
round(calc_credible_intervals(bart_machine_cv, X[1:2, ]), 3)
pd_plot(bart_machine_cv, j = "glu")