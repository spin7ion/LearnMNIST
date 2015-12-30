### Contains data loading utility functions
source("load_data.R")

#data <- loadMNISTData("train-images-idx3-ubyte", "train-labels-idx1-ubyte");
data <- loadMNISTData("./exdb/train-images.idx3-ubyte", "./exdb/train-labels.idx1-ubyte")
trainLabels <- data$labels
trainData <- data$data

print(dim(trainData))
print(dim(trainLabels))


#data <- loadMNISTData("t10k-images-idx3-ubyte", "t10k-labels-idx1-ubyte");
data <- loadMNISTData("./exdb/t10k-images.idx3-ubyte", "./exdb/t10k-labels.idx1-ubyte")
testLabels <- data$labels
testData <- data$data



#Lets first normalize the data and drop features with maximal value 0
trainDataMax <- apply(trainData, max, MARGIN = c(2))

#With a normalize function we will normalize our dataset
normalize <- function(train, max) {
  train[, max > 1] / max[max > 1]
}

#Applying Normalized functions
trainDataNormalized <- normalize(trainData, trainDataMax)
testDataNormalized <- normalize(testData, trainDataMax)

#Define the predictor variables with our normalized data
X <- as.matrix(trainDataNormalized)

#Define a response variable
Y <- as.matrix(trainLabels)

#As logistic regression is a binary classifier, we implement it to classify only two numbers
xzero <- X[Y==0,]
xone  <-X[Y==1,]
yzero <- Y[Y==0]
yone <- Y[Y==1]
#We will now concatenate both matrices so we can feed them into our classifier
xbinary <- rbind(xzero, xone)
ybinary <- c(yzero,yone)



#Add ones to our X to vectorize and act as intercept terms
X1 <- cbind(rep(1,nrow(xbinary)),xbinary)
X1Size <- nrow(X1)


#Lets define the sigmoid function
sigmoid <- function(x)
{
  g <- 1/(1+exp(-x))
  return(g)
}

#We need a cost function that is convex 
costJwithReg <- function(theta)
{
  lambda <- 1
  m <- X1Size 
  hypothesis <- sigmoid(X1%*%theta)
  J <- (1/m)*sum((-ybinary*log(hypothesis)) - ((1-ybinary)*log(1-hypothesis)))
  costRegularizationTerm <- lambda/(2*m) * sum(theta^2)
  Jreg <- J+costRegularizationTerm 
  return(J)
}

#We define an initial theta as a very small randomized value
initial_theta <- rep(runif(1,0,1)*0.001,ncol(X1))

#Cost at inital theta
cost_theta <- costJwithReg(initial_theta)
cat("Cost theta: ", cost_theta)

# We derive the theta using gradient descent with the built-in optim function from R
theta_optim <- optim(par=initial_theta,fn=costJwithReg)

#Lets set the theta
theta_par <- theta_optim$par

#Cost at optimal value of the theta
cost_theta_optim <- theta_optim$value
cat("Theta Optim Cost: ", cost_theta_optim)

#We evaluate the delta in cost
delta_cost <- cost_theta_optim - cost_theta;
cat("Delta Cost: ", delta_cost)

# Lets evaluate with a test dataset and lets vectorize it
# First we normalize it and then we select only two classes
testDataNormMatrix <- as.matrix(testDataNormalized)
Z <- as.matrix(testDataNormMatrix)
zzero <- Z[testLabels==0,]
zone  <-Z[testLabels==1,]
testLabelsZero <- testLabels[testLabels==0]
testLabelsOne <- testLabels[testLabels==1]
#We will now concatenate both matrices so we can feed them into our classifier
zBinary <- rbind(zzero, zone)
testLabelsBinary <- c(testLabelsZero,testLabelsOne)

Z1 <- cbind(rep(1,nrow(zBinary)),zBinary)
Z1T <- t(Z1)

prob <- sigmoid(Z1%*%theta_par)

#We define a function to evaluate the accuracy of our classifier
binary_classifier_accuracy <- function(theta, X,y){
  correct <- sum(y == (sigmoid(X%*%theta) > 0.5))
  accuracy <- correct / length(y)
  return(accuracy)
  
}
bca_train <- binary_classifier_accuracy(theta_par, X1, ybinary)
cat("Accuracy on training data: ", bca_train)

bca_train <- binary_classifier_accuracy(theta_par, Z1, testLabelsBinary)
cat("Accuracy on test data: ", bca_train)

