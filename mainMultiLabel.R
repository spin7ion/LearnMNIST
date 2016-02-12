### Clear workspace
rm(list = ls()) 
### Load utility sources
source("load_data.R") #loader helper
source("showDataImg.R") #helper function to show dataimg with label

#load training data
data <- loadMNISTData("./exdb/train-images.idx3-ubyte", "./exdb/train-labels.idx1-ubyte")
trainLabels <- data$labels
trainData <- data$data

#load test data
data <- loadMNISTData("./exdb/t10k-images.idx3-ubyte", "./exdb/t10k-labels.idx1-ubyte")
testLabels <- data$labels
testData <- data$data

### Normalize data
#get max value of pixel value
pixelMax<-max(trainData)
trainDataNormalized<-trainData/pixelMax
testDataNormalized<-testData/pixelMax

#Define the predictor variables with our normalized data
X <- as.matrix(trainDataNormalized)

#Define a response variable
Y <- as.matrix(trainLabels)

### Prepare test data
Z <- as.matrix(testDataNormalized)
Z1 <- cbind(rep(1,nrow(Z)),Z)

# make some garbage collection to optimize memory consumption
rm(testData,testDataNormalized,trainData,trainDataNormalized,trainLabels,Z,data,pixelMax)

getTrainingDataForDigit<-function(digit){

  #We will now concatenate both matrices so we can feed them into our classifier
  xbinary <- rbind( X[Y==digit,], X[Y!=digit,])
  ybinary <- sign(c(sign(Y[Y==digit]+10),sign(Y[Y!=digit]-10)+1))
  
  #Add ones to our X to vectorize and act as intercept terms
  X1 <- cbind(rep(1,nrow(xbinary)),xbinary)#1 is inserted because of theta0
  
  return(list("X1" = X1, "ybinary" = ybinary))
}

### moreThanZero
moreThanZero <- function(z){
  return(pmax(0,z))
}

###moreThan zero derivative
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

### Create cost function derivative
dJwithReg <- function(X1,ybinary,theta)
{
  X1Size <- nrow(X1)
  hyperplane<-X1%*%theta
  dJ <- (1/X1Size)*colSums( -X1*ybinary*dMoreThanZero(1-hyperplane)+(1-ybinary)*X1*dMoreThanZero(1+hyperplane) )
  return(dJ)
}

grad.descent <- function(trainDigitData,theta, maxit){
  alpha = .05 # set learning rate
  for (i in 1:maxit) {
    if(i%%10==0){
      cat("iteration:",i,"Cost theta: ", JwithReg(trainDigitData$X1,trainDigitData$ybinary,theta),"\n")
    }
    theta <- theta - alpha * dJwithReg(trainDigitData$X1,trainDigitData$ybinary,theta)
  }
  return(theta)
}

trainSVM<-function(){
  trainedTheta<-NULL
  for(i in 0:9) {
    cat("training digit:",i,"\n")
    trainDigitData <- getTrainingDataForDigit(i)
    initial_theta <-runif(ncol(trainDigitData$X1), min = -1, max = 1)
    theta_par <- grad.descent(trainDigitData,initial_theta,50)
    trainedTheta<-cbind(trainedTheta,theta_par)
    cat("Done\n")
  }
  return(trainedTheta)
}

trainedThetaPars<-trainSVM()

predictedLabels<-max.col(Z1%*%trainedThetaPars)-1 #subtract 1 because we have 0:9 range and not 1:10

cat("predicted:",100*sum(predictedLabels==testLabels)/nrow(testLabels),"%")

predictMatrix<-matrix(0,10,10)

for(i in 1:nrow(testLabels)){
  predictMatrix[[ predictedLabels[[i]]+1,testLabels[[i]]+1 ]]<-predictMatrix[[ predictedLabels[[i]]+1,testLabels[[i]]+1 ]]+1
}
View(predictMatrix)

