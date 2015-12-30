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
yzero <- Y[Y==0]
yone <- Y[Y==1]

#We will now concatenate both matrices so we can feed them into our classifier
xbinary <- rbind(xzero, xone)
ybinary <- c(yzero,yone)

#Add ones to our X to vectorize and act as intercept terms
X1 <- cbind(rep(1,nrow(xbinary)),xbinary)#1 is inserted because of theta0
X1Size <- nrow(X1)

### moreThanZero
moreThanZero <- function(z){
  return(min(max(0,sign(z)),1))
}

### Create cost function
JwithReg <- function(theta)
{
  lambda <- 0.0  # regularization of trade-off.
  mu <- 0.1 # learning rate
  hyperplane<-X1%*%theta
  J <- -(1/mu)*sum( ybinary*moreThanZero(1-hyperplane)+(1-ybinary)*(hyperplane+1) )+(lambda/mu) * sum(theta^2)
  return(J)
}

#We define an initial theta as a very small randomized value
initial_theta <- rep(runif(1,0,1)*0.001,ncol(X1))

#Cost at inital theta
cost_theta <- JwithReg(initial_theta)
cat("Cost theta: ", cost_theta)

# We derive the theta using gradient descent with the built-in optim function from R
theta_optim <- optim(par=initial_theta,fn=JwithReg)
#Lets set the theta
theta_par <- theta_optim$par

#Cost at optimal value of the theta
cost_theta_optim <- theta_optim$value
cat("Theta Optim Cost: ", cost_theta_optim)

#We evaluate the delta in cost
delta_cost <- cost_theta_optim - cost_theta;
cat("Delta Cost: ", delta_cost)
### TEST ON some data
sign(theta_par%*%c(1,xbinary[12660,]))
sign(theta_par%*%c(1,xbinary[10,]))
