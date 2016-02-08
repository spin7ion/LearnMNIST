### Clear workspace
rm(list = ls()) 
### Load utility sources
source("load_data.R") #loader helper
source("showDataImg.R") #helper function to show dataimg with label

#load training data
data <- loadMNISTData("./exdb/train-images.idx3-ubyte", "./exdb/train-labels.idx1-ubyte")
trainLabels <- data$labels
trainData <- data$data

print(dim(trainData))
print(dim(trainLabels))
#showDataImg(trainData,trainLabels,10)

#load test data
data <- loadMNISTData("./exdb/t10k-images.idx3-ubyte", "./exdb/t10k-labels.idx1-ubyte")
testLabels <- data$labels
testData <- data$data

print(dim(testData))
print(dim(testLabels))

### Normalize data
#get max value of pixel value
pixelMax<-max(trainData)
trainDataNormalized<-trainData/pixelMax

#Define the predictor variables with our normalized data
X <- as.matrix(trainDataNormalized)

#Define a response variable
Y <- as.matrix(trainLabels)

getTrainingDataForDigit<-function(digit){
  xDigit<- X[Y==digit,]
  xNotDigit<- X[Y!=digit,]
  
  yDigit <-  Y[Y==digit]+1
  yDigit <- yDigit/yDigit
  yNotDigit <- Y[Y!=digit]+1
  yNotDigit <- yNotDigit/yNotDigit-1
  
  #We will now concatenate both matrices so we can feed them into our classifier
  xbinary <- rbind(xDigit, xNotDigit)
  ybinary <- sign(c(yDigit,yNotDigit))
  
  #Add ones to our X to vectorize and act as intercept terms
  X1 <- cbind(rep(1,nrow(xbinary)),xbinary)#1 is inserted because of theta0
 
  return(list("X1" = X1, "ybinary" = ybinary))
}

### moreThanZero
moreThanZero <- function(z){
  return(pmax(0,z))
}
dMoreThanZero <- function(z){
  return(pmin(pmax(0,sign(z)),1))
}

### Create cost function
JwithReg <- function(X1,ybinary,theta)
{
  X1Size <- nrow(X1)
  hyperplane<-X1%*%theta
  J <- (1/X1Size)*sum( ybinary*moreThanZero(1-hyperplane)+(1-ybinary)*moreThanZero(hyperplane+1) )
  return(J)
}

### Create cost function
dJwithReg <- function(X1,ybinary,theta)
{
  X1Size <- nrow(X1)
  hyperplane<-X1%*%theta
  dJ <- (1/X1Size)*colSums( -X1*ybinary*dMoreThanZero(1-hyperplane)+(1-ybinary)*X1*dMoreThanZero(1+hyperplane) )
  return(dJ)
}

grad.descent <- function(X1,ybinary,theta, maxit){
  alpha = .05 # set learning rate
  for (i in 1:maxit) {
    if(i%%10==0){
      cost_theta <- JwithReg(X1,ybinary,theta)
      cat("iteration:",i,"Cost theta: ", cost_theta)
      cat("\n")
    }
    theta <- theta - alpha * dJwithReg(X1,ybinary,theta)  
    
  }
  return(theta)
}

trainSVM<-function(){
  trainedTheta<-list()
  for(i in 0:9) {
    cat("training digit:",i,"\n")
    trainDigitData <- getTrainingDataForDigit(i)
    initial_theta <-runif(ncol(trainDigitData$X1), min = -1, max = 1)
    theta_par <- grad.descent(trainDigitData$X1,trainDigitData$ybinary,initial_theta,50)
    trainedTheta[[i+1]]<-theta_par
  }
  return(trainedTheta)
}
trainedThetaPars<-trainSVM()

predictLabel<-function(xtest,theta_pars_list){
  minCost<- -Inf
  minI=NaN
  
  for(i in 0:9) {
    currentCost<-theta_pars_list[[i+1]]%*%xtest;
    #cat("i:",i,"cost:",currentCost,"\n")
    if(minCost<currentCost){
      minCost<-currentCost
      minI<-i
    }
  }
  return(minI);
}
predictLabel(c(1,xone[4,]),trainedThetaPars)

X1 <- cbind(rep(1,nrow(X)),X)

costs<-X1%*%trainedThetaPars[[1]]
predictedLabels<-rep(0,nrow(costs));
doNotCount<-0
for(i in 0:9) {
  currentCosts<-X1%*%trainedThetaPars[[i+1]]
  for(j in 1:nrow(costs)){
    if(costs[j]<currentCosts[j] && i!=doNotCount){
      costs[j]<-currentCosts[j]
      predictedLabels[j]<-i
    }
  }
}

successfulPredicts<-0
totalPredicts<-0


for(i in 1:60000){
  if(predictedLabels[i]==Y[i] && Y[i]!=doNotCount){
    successfulPredicts<-successfulPredicts+1
  }  
  if(Y[i]!=doNotCount){
    totalPredicts<-totalPredicts+1
  }
}

cat("predicted:",100*successfulPredicts/totalPredicts,"%")

#We define an initial theta as a very small randomized value
#initial_theta <- rep(runif(1,0,1)*0.001,ncol(X1))

#Cost at inital theta
#cost_theta <- JwithReg(theta_par)
#cat("Cost theta: ", cost_theta)

# We derive the theta using gradient descent with the built-in optim function from R
#theta_optim <- optim(par=initial_theta,fn=JwithReg)
#Lets set the theta
#theta_par <- theta_optim$par

#Cost at optimal value of the theta
#cost_theta_optim <- theta_optim$value
#cat("Theta Optim Cost: ", cost_theta_optim)

#We evaluate the delta in cost
#delta_cost <- cost_theta_optim - cost_theta;
#cat("Delta Cost: ", delta_cost)
### TEST ON some data

#max(0,sign(theta_par%*%c(1,xzero[1,])))

#max(0,sign(theta_par%*%c(1,xnine[22,])))



#predictedLabels <- pmax(0,sign(Z1%*%theta_par))

#We define a function to evaluate the accuracy of our classifier
#binary_classifier_accuracy <- function(theta, X,y){
  #sign(theta_par%*%c(1,xnine[12,]))
# correct <- sum( y == max(0,sign(X%*%theta)) )
  
# accuracy <- correct / length(y)
# return(accuracy)
#}



###Check accuracy of classifier
#bca_train <- binary_classifier_accuracy(theta_par, X1, ybinary)
#cat("Accuracy on training data: ", bca_train*100,"%")

#bca_train <- binary_classifier_accuracy(theta_par, Z1, testLabelsBinary)
#cat("Accuracy on test data: ", bca_train*100,"%")

#TP <- sum((testLabelsBinary == 1) * (predictedLabels == 1))  # True Positive
#TN <- sum((testLabelsBinary == 0) * (predictedLabels == 0))  # True Negative
#FP <- sum((testLabelsBinary == 0) * (predictedLabels == 1))  # False Positive
#FN <- sum((testLabelsBinary == 1) * (predictedLabels == 0))  # False Negative

#accuracy <- (TP + TN) / (TP + TN + FP + FN)
#precision <- TP / (TP + FP)
#recall <- TP / (TP + FN)
#specificity <- TN / (TN + FN)
#fmeasure <- 2 * TP / (2 * TP + FN + FP)
#FDR <- FP / (FP + TP)

#cat("Accuracy: ", accuracy)
#cat("Precision: ", precision)
#cat("Recall: ", recall)
#cat("Specificity: ", specificity)
#cat("Fmeasure: ", fmeasure)
#cat("FDR: ", FDR)

