db = read.csv("http://www.rob-mcculloch.org/data/diabetes.csv")

xB = as.matrix(db[,2:11]) #x for BART
yBL = db$y

rmsef = function(y,yhat) {return(sqrt(mean((y-yhat)^2)))}
nd=30 #number of train/test splits
n = length(yBL)
ntrain=floor(.75*n) #75% in train each time
rmseB = rep(0,nd) #BART rmse
fmatB=matrix(0.0,n-ntrain,nd) #out-of-sample BART predictions


for(i in 1:nd) {
  set.seed(i)
  # train/test split
  ii=sample(1:n,ntrain)
  #y
  yTrain = yBL[ii]; yTest = yBL[-ii]
  #x for BART
  xBTrain = xB[ii,]; xBTest = xB[-ii,]

  #BART
  bfTrain = wbart(xBTrain,yTrain,xBTest)
  #get predictions on test
  yBhat = bfTrain$yhat.test.mean

  #store results
  rmseB[i] = rmsef(yTest,yBhat); 
  fmatB[,i]=yBhat;
}




bfd = wbart(xB,yBL,nskip=1000,ndpost=5000) #keep it simple
plot(bfd$yhat.train.mean,yBL)
abline(0,1,col="red",lty=2)