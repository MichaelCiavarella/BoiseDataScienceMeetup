# Data from Iris Data Set
# Random Forest Classification Example
# Randall Shane, PhD
# 2 Mar 2016

import random

from data import data
from sklearn. ensemble import RandomForestClassifier

# Step 1: Create train and test sets
train = [data[x] for x in sorted(random.sample(xrange(len(data)), 100))]
train_refs = [x[0] for x in train]
test = [x for x in data if x[0] not in train_refs]

# Step 2: Train the algorithms
predictor = [x[1:5] for x in train]
classes = [x[5] for x in train]

rf = RandomForestClassifier(n_estimators=750,
                            n_jobs=-1,
                            criterion='gini',
                            max_depth=None,
                            min_samples_split=2,
                            min_samples_leaf=1,
                            min_weight_fraction_leaf=0.0,
                            max_features='auto',
                            max_leaf_nodes=None,
                            bootstrap=True,
                            oob_score=False,
                            random_state=None,
                            verbose=0,
                            warm_start=False,
                            class_weight=None)

fit = rf.fit(predictor, classes)

# Side Step: Attribute Importance ???
var_importance = rf.transform(predictor, threshold=None)


# Step 3: Test set probability and prediction
test_predictor = [x[1:5] for x in test]
test_classes = [x[5] for x in test]

# Provides probability of belonging to each class
rf_prob = rf.predict_proba(test_predictor).tolist()
# Predicts the class
rf_cls = rf.predict(test_predictor).tolist()


# Step 4: Evaluating Accuracy
pop = len(data)
error = 0
for i, c in enumerate(rf_cls):
    if c != test_classes[i]:
        error += 1

error_rate = round((float(error) / float(pop)) * 100, 2)
accuracy = round((float(pop - error) / float(pop)) * 100, 2)

# error_rate  1.33
# accuracy   98.67
