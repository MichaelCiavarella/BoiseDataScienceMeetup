# Finding Outliers
# Boise Data Science Meetup
# June 2016
# Randall Shane, PhD

rm(list=ls(all.names=TRUE))
options(digits=4, warn=-1)

# Package Installs
install.packages('Rlof', dependencies=TRUE)
install.packages('outliers', dependencies=TRUE)
install.packages('DMwR', dependencies=TRUE)

# Create a vector of random data
x <- runif(1000000, min=0, max=100)
x[x < 1] <- NA
x[x > 99] <- 0
x <- c(x, runif(100, min=-23, max=0))
x <- c(x, runif(100, min=103, max=1234))
x <- x[201:1000200]
length(x)

# Inferential Statistics
length(x)
mean(x)
median(x)
var(x)
sd(x)
min(x)
max(x)

# Nulls & Zeros
length(x[is.na(x)])
length(x[x == 0])

# Histogram
hist(x,
     include.lowest = TRUE, density = 10,
     main = 'Histogram of x', col = 'blue',
     xlab = 'x values', axes = TRUE,
     plot = TRUE,  
     # plot = FALSE  # list of breaks and counts is returned
     warn.unused = TRUE)

# ECDF Plot
x.ecdf = ecdf(x)

require(graphics)
plot(x.ecdf,
     main = 'ECDF of x',
     col = 'blue',
     xlab = 'x value')

# Density
library(stats)
# NOTE: must remove NA
x.density = density(x)
x.density
plot(x.density,
     main = 'Density of x',
     col = 'blue')


# 3 Sigma Edit rule
mean(x)
sd(x)
length(x[x > mean(x) + (3 * sd(x))])


#   Outliers Package/Library
library(outliers)
# Chi-squared test
chisq.out.test(x, variance=var(x), opposite = FALSE)
# Grubbs test
grubbs.test(x, type = 10, opposite = FALSE, two.sided = FALSE)
# Dixon test (samples only)
dixon.test(sample(x, 30), type = 0, opposite = FALSE, two.sided = TRUE)
# outlier -> Finds value with largest difference between it and sample mean
outlier(x, opposite = FALSE, logical = FALSE)
# scores -> This function calculates normal, t, chi-squared, IQR and MAD scores of given data.
scores <- scores(x, type = "iqr", prob = 1, lim = NA)
# type = c("z", "t", "chisq", "iqr", "mad")
tail(sort(scores), 10)  # provides a vector of n high values


# DMwR outlier.scores
library(DMwR)
# NOTE: must remove NA & 0
outlier.scores <- lofactor(x, k=5)
plot(density(outlier.scores))

# choose top n outliers
outliers <- order(outlier.scores, decreasing=T)[1:5]
outliers


# Removing outliers and prepping data
x <- x[! is.na(x)]  # nulls
x <- x[! x == 0]  # zeros
x <- x[! x > mean(x) + (3 * sd(x))] # 3 Sigma
length(x)



