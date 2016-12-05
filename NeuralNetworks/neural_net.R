# Neural Networks
# Boise Data Science Meetup
# December 2016
# Randall Shane, PhD

# Predict the median value of owner-occupied homes
# using the Boston data set from the MASS library

rm(list=ls(all.names=TRUE))
install.packages('MASS', dependencies=TRUE)

library(MASS)
data <- Boston
set.seed(500)
head(data)

# Check for missing data
apply(data,2,function(x) sum(is.na(x)))

# Split test and training set
index <- sample(1:nrow(data),round(0.75*nrow(data)))
train <- data[index,]
test <- data[-index,]

# Create a linear regression model (glm) on the test set
lm.fit <- glm(medv~., data=train)
summary(lm.fit)
pr.lm <- predict(lm.fit,test)
MSE.lm <- sum((pr.lm - test$medv)^2)/nrow(test)

# Preprocessing using normalization
maxs <- apply(data, 2, max) 
mins <- apply(data, 2, min)
scaled <- as.data.frame(scale(data, center = mins, scale = maxs - mins))
train_ <- scaled[index,]
test_ <- scaled[-index,]

# Fit the neural net
install.packages('neuralnet', dependencies=TRUE)
library(neuralnet)

n <- names(train_)
f <- as.formula(paste("medv ~", 
                      paste(n[!n %in% "medv"], 
                            collapse = " + ")))
nn <- neuralnet(f, 
                data=train_,
                hidden=c(5,3),
                linear.output=T)
plot(nn)

# Predict the values for the test set
pr.nn <- compute(nn,test_[,1:13])
pr.nn_ <- pr.nn$net.result*(max(data$medv)-min(data$medv))+min(data$medv)
test.r <- (test_$medv)*(max(data$medv)-min(data$medv))+min(data$medv)
MSE.nn <- sum((test.r - pr.nn_)^2)/nrow(test_)

# Check MSE
# linear model
MSE.lm
# neural net
MSE.nn

# PLot the MSE
plot(test$medv,pr.nn_,col='red',main='Real vs predicted NN',pch=18,cex=0.7)
points(test$medv,pr.lm,col='blue',pch=18,cex=0.7)
abline(0,1,lwd=2)
legend('bottomright',legend=c('NN','LM'),pch=18,col=c('red','blue'))
