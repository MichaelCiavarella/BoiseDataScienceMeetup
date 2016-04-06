# k-Means Example
# Boise Data Science Meetup
# 5 April 2016
# Randall Shane, PhD

rm(list=ls(all.names=TRUE))
options(warn=-1)

# Step 1: Prep Data
df <- read.csv(file="crime_data.csv",head=TRUE,sep=",")
dim(df)
head(df)

df <- na.omit(df)  # omit NA's
df <- data.frame(df)  # data frame


# Step 2: Elbow Bend Analysis
mydata <- df[,2:5]
mydata
wss <- (nrow(mydata)-1)*sum(apply(mydata,2,var))
for (i in 2:15) wss[i] <- sum(kmeans(mydata,centers=i)$withinss)
plot(1:15, 
     wss, 
     type="b", 
     main = "Elbow Bend Analysis",
     xlab = "Number of Clusters",
     ylab = "Sum of Squares")


# Step 3: Clustering - 5 clusters
fit <- kmeans(mydata, # data
              centers = 5,     # clusters desired
              iter.max = 1000, # max iterations
              nstart = 25,     # initial configs
              # algorithm = "Lloyd", "Forgy" # original
              algorithm = "Hartigan-Wong" # efficient
              # algorithm = "MacQueen"    # centroid updates
              )


# Step 4: Output
fit
fit[1]  # clusters
fit[2]  # centers
fit[3]  # total sum of squres
fit[4]  # within sum of squares (each)
fit[5]  # total within sum of squares
fit[6]  # between sum of squares
fit[7]  # size of clusters
fit[8]  # iterations
fit[9]

# Bind cluster to original data
df <- cbind(df, fit[1])

# Sort by cluster
df <- df[with(df, order(cluster)), ]
df
