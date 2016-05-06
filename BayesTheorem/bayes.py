# Bayes Naive Classifier
# Academic Example
# Randall Shane, PhD
# 6 May 2016

from sklearn import datasets
from sklearn.naive_bayes import GaussianNB

iris = datasets.load_iris()

clf = GaussianNB()
fit = clf.fit(iris.data, iris.target)

pred = clf.predict(iris.data)

correct = (iris.target == pred).sum()
errors = (iris.target != pred).sum()

print "Errors: %i of of %i" % (errors, len(pred))
print "Pct Correct: ", float(correct) / len(pred)
