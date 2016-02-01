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
showDataImg(trainData,trainLabels,10)

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

###Divide by 2 classes
xzero <- X[Y==0,]
xone  <-X[Y==1,]
xtwo  <-X[Y==2,]
xthree  <-X[Y==3,]
xfour  <-X[Y==4,]
xfive  <-X[Y==5,]
xsix  <-X[Y==6,]
xseven  <-X[Y==7,]
xeight  <-X[Y==8,]
xnine  <-X[Y==9,]

yzero <- Y[Y==0]
yone <- Y[Y==1]
ytwo <- Y[Y==2]
ythree <- Y[Y==3]
yfour <- Y[Y==4]
yfive <- Y[Y==5]
ysix <- Y[Y==6]
yseven <- Y[Y==7]
yeight <- Y[Y==8]
ynine <- Y[Y==9]

#We will now concatenate both matrices so we can feed them into our classifier
xbinary <- rbind(xeight, xthree)
ybinary <- sign(c(yeight,ythree))

#Add ones to our X to vectorize and act as intercept terms
X1 <- cbind(rep(1,nrow(xbinary)),xbinary)#1 is inserted because of theta0
X1Size <- nrow(X1)

### moreThanZero
moreThanZero <- function(z){
  return(pmax(0,z))
}
dMoreThanZero <- function(z){
  return(pmin(pmax(0,sign(z)),1))
}

### Create cost function
JwithReg <- function(theta)
{
  hyperplane<-X1%*%theta
  J <- (1/X1Size)*sum( ybinary*moreThanZero(1-hyperplane)+(1-ybinary)*moreThanZero(hyperplane+1) )
  return(J)
}



### Create cost function
dJwithReg <- function(theta)
{
  hyperplane<-X1%*%theta
  dJ <- (1/X1Size)*colSums( -X1*ybinary*dMoreThanZero(1-hyperplane)+(1-ybinary)*X1*dMoreThanZero(1+hyperplane) )
  return(dJ)
}

grad.descent <- function(theta, maxit){
  alpha = .05 # set learning rate
  maxit = 200
  for (i in 1:maxit) {
    cost_theta <- JwithReg(theta)
    cat("iteration:",i,"Cost theta: ", cost_theta)
    theta <- theta - alpha * dJwithReg(theta)  
    cat("\n")
  }
  return(theta)
}

#We define an initial theta as a very small randomized value
#initial_theta <- rep(runif(1,0,1)*0.001,ncol(X1))
initial_theta <-runif(ncol(X1), min = -1, max = 1)
theta_par <-grad.descent(initial_theta)
#Cost at inital theta
cost_theta <- JwithReg(theta_par)
cat("Cost theta: ", cost_theta)

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

max(0,sign(theta_par%*%c(1,xzero[1,])))

max(0,sign(theta_par%*%c(1,xnine[22,])))


### Prepare test data
#normalize
testDataNormalized<-testData/pixelMax
Z <- as.matrix(testDataNormalized)
#test only on zeros and ones
zzero <- Z[testLabels==0,]
zone  <-Z[testLabels==1,]
ztwo  <-Z[testLabels==2,]
zthree  <-Z[testLabels==3,]
zfour  <-Z[testLabels==4,]
zfive  <-Z[testLabels==5,]
zsix  <-Z[testLabels==6,]
zseven  <-Z[testLabels==7,]
zeight  <-Z[testLabels==8,]
znine  <-Z[testLabels==9,]

testLabelsZero <- testLabels[testLabels==0]
testLabelsOne <- testLabels[testLabels==1]
testLabelsTwo <- testLabels[testLabels==2]
testLabelsThree <- testLabels[testLabels==3]
testLabelsFour <- testLabels[testLabels==4]
testLabelsFive <- testLabels[testLabels==5]
testLabelsSix <- testLabels[testLabels==6]
testLabelsSeven <- testLabels[testLabels==7]
testLabelsEight <- testLabels[testLabels==8]
testLabelsNine <- testLabels[testLabels==9]
#Concatenate to feed classifier
zBinary <- rbind(zeight, zthree)
testLabelsBinary <- sign(c(testLabelsEight,testLabelsThree))
#Add ones
Z1 <- cbind(rep(1,nrow(zBinary)),zBinary)

predictedLabels <- pmax(0,sign(Z1%*%theta_par))

#We define a function to evaluate the accuracy of our classifier
binary_classifier_accuracy <- function(theta, X,y){
  #sign(theta_par%*%c(1,xnine[12,]))
  correct <- sum( y == max(0,sign(X%*%theta)) )
  
  accuracy <- correct / length(y)
  return(accuracy)
}


  
###Check accuracy of classifier
bca_train <- binary_classifier_accuracy(theta_par, X1, ybinary)
cat("Accuracy on training data: ", bca_train*100,"%")

bca_train <- binary_classifier_accuracy(theta_par, Z1, testLabelsBinary)
cat("Accuracy on test data: ", bca_train*100,"%")

TP <- sum((testLabelsBinary == 1) * (predictedLabels == 1))  # True Positive
TN <- sum((testLabelsBinary == 0) * (predictedLabels == 0))  # True Negative
FP <- sum((testLabelsBinary == 0) * (predictedLabels == 1))  # False Positive
FN <- sum((testLabelsBinary == 1) * (predictedLabels == 0))  # False Negative

accuracy <- (TP + TN) / (TP + TN + FP + FN)
precision <- TP / (TP + FP)
recall <- TP / (TP + FN)
specificity <- TN / (TN + FN)
fmeasure <- 2 * TP / (2 * TP + FN + FP)
FDR <- FP / (FP + TP)

cat("Accuracy: ", accuracy)
cat("Precision: ", precision)
cat("Recall: ", recall)
cat("Specificity: ", specificity)
cat("Fmeasure: ", fmeasure)
cat("FDR: ", FDR)

