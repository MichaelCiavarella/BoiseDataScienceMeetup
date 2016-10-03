# Nearest Neighbor Classifiers
# October 2016

rm(list=ls(all.names=TRUE))
options(digits=7)
options(warn=-1)

# Step 1:  Load Data
iris <- read.csv(url("http://archive.ics.uci.edu/ml/machine-learning-databases/iris/iris.data"), header = FALSE)
names(iris) <- c("Sepal.Length", "Sepal.Width", "Petal.Length", "Petal.Width", "Species")
head(iris)

# Step 2: Visualize your data

install.packages('ggvis', dependencies=TRUE)
library(ggvis)

# sepal length & width
iris %>% ggvis(~Sepal.Length, 
               ~Sepal.Width, 
               fill = ~Species) %>% layer_points()

# petal length & width
iris %>% ggvis(~Petal.Length, 
               ~Petal.Width, 
               fill = ~Species) %>% layer_points()

summary(iris)
head(iris)


# Step 3: Library, Training & Test Sets

library(class)  # activiate built in library

# create training (67%) and test(33%) sets
set.seed(1234)
ind <- sample(2, nrow(iris), 
              replace=TRUE, 
              prob=c(0.67, 0.33))
iris.training <- iris[ind==1, 1:4]
iris.test <- iris[ind==2, 1:4]
nrow(iris.training) # 110
nrow(iris.test) # 40

# store class labels
iris.trainLabels <- iris[ind==1, 5]
iris.testLabels <- iris[ind==2, 5]


# Step 4: Create 'model' and Predict
iris_pred <- knn(train = iris.training, 
                 test = iris.test, 
                 cl = iris.trainLabels, 
                 k=3)  # This sets the value of k

# results - these aren;t very useful like this!
iris_pred

cbind(iris_pred, iris.testLabels)


# Step 5: Evaluation
install.packages('gmodels', dependencies=TRUE)
library(gmodels)

CrossTable(x = iris.testLabels, y = iris_pred, prop.chisq=FALSE)


