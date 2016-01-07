# Gender Inference Model
# Academic Example
# Randall Shane, PhD
# 6 Jan 2015

import numpy
import pymongo
import random
import time

from names import female_names, male_names
from sklearn.ensemble import RandomForestClassifier
from sklearn.neighbors import KNeighborsClassifier
from sklearn.linear_model import SGDClassifier


# Architecture, DB & Setup
conn = pymongo.MongoClient()
db = conn['infer']
data = db['data']
gnd_test = db['multi_test']
gnd_train = db['multi_train']


referrer_lookup = {'': 0,
                   'Direct Nav': 1,
                   'SEO': 2,
                   'Social Media': 3,
                   'E-Mail': 4,
                   'Non Paid Referral': 5,
                   'Affiliate': 6,
                   'E-mail': 7,
                   'SEM / CPC': 8,
                   'Military': 9,
                   'Banner': 10,
                   'Sponsorship': 11,
                   'Other': 0,
                   'Print': 13,
                   'television': 14,
                   'Public Relations': 15,
                   'Product Feed': 16,
                   'Display': 17,
                   'Social media': 3,
                   'PR': 15,
                   'DC': 16,
                   'Unknown': 0,
                   'print': 13,
                   'Widget': 18,
                   '3rd Party Email Blast': 19,
                   'Direct Mail': 20}


def prep_data(sample):
    start = time.time()
    print 'Step 1) Reading Data'
    # Derive Data
    records = [x for x in data.find({'Gender': {'$lt': 2}},
                                    {'_id': 1,
                                        'First_Name': 1,
                                        'Goal': 1,
                                        'Gender': 1,
                                        'Referrer_Channel': 1,
                                        'Total_Site_Hits': 1,
                                        'Use_0': 1,
                                        'Use_1': 1,
                                        'Use_2': 1,
                                        'Use_3': 1,
                                        'Use_4': 1,
                                        'Use_5': 1,
                                        'Use_6': 1,
                                        'Use_7': 1,
                                        'Use_8': 1,
                                        'Use_9': 1,
                                        'Use_10': 1,
                                        'Total_Forum_Hits': 1,
                                        'Use_11': 1}).limit(sample)]

    # Clean Existing Collections
    gnd_test.drop()
    gnd_train.drop()

    print '\tRead time: %f seconds' % (time.time() - start)

    # Gender Data Prep
    # Loop, transform and write test and train collections
    print 'Step 2) Performing Transform and Save'
    for i, r in enumerate(records):
        if i < sample:
            # Start massive validity check
            if r['Total_Site_Hits'] != ''\
                    and r['Use_0'] != '' and r['Use_1'] != '' and r['Use_2'] != ''\
                    and r['Use_3'] != '' and r['Use_4'] != '' and r['Use_5'] != ''\
                    and r['Use_6'] != '' and r['Use_7'] != '' and r['Use_8'] != ''\
                    and r['Use_9'] != '' and r['Use_10'] != '' and r['Total_Forum_Hits'] != '' and r['Use_11'] != '':

                # Then transform
                if r['First_Name'] in male_names or r['First_Name'] in female_names:
                    r['Common_Name'] = 1
                else:
                    r['Common_Name'] = 0
                r['referrer'] = r['Referrer_Channel']

                ind_data = [float(r['Common_Name']),
                            float(r['referrer']),
                            int(r['Total_Site_Hits']),
                            float(r['Use_0']),
                            float(r['Use_1']),
                            float(r['Use_2']),
                            float(r['Use_3']),
                            float(r['Use_4']),
                            float(r['Use_5']),
                            float(r['Use_6']),
                            float(r['Use_7']),
                            float(r['Use_8']),
                            float(r['Use_9']),
                            float(r['Use_10']),
                            float(r['Total_Forum_Hits']),
                            float(r['Use_11'])]

                # Test or Train ?
                decider = random.randint(1, 10)
                if decider % 2 == 0:
                    gnd_test.insert({'_id': r['_id'],
                                     'ind': ind_data,
                                     'gender': r['Gender']})
                else:
                    gnd_train.insert({'_id': r['_id'],
                                      'ind': ind_data,
                                      'gender': r['Gender']})

    print '\tPrep time: %f seconds' % (time.time() - start)


def classifiers():
    print 'Step 3) Starting Classification'
    start = int(time.time())
    # rand_smpl = [ mylist[i] for i in sorted(random.sample(xrange(len(mylist)), 4)) ]
    train = [x for x in gnd_train.find({},
                                       {'_id': 0,
                                        'ind': 1,
                                        'gender': 1})]

    predictor = [x['ind'] for x in train]
    classes = [x['gender'] for x in train]

    print '\tTraining the algorithms:'

    # Random Forest Classifier
    print '\t\tRandom Forest'
    rf_clf = RandomForestClassifier(n_estimators=750, n_jobs=-1)
    rf_clf.fit = rf_clf.fit(predictor, classes)

    # Attribute Importance
    # rf_vi = rf_clf.transform(predictor, threshold=None)
    # print 'Random Forest Attribute Importance: ', rf_vi

    # Stochastic Gradient Descent
    print '\t\tStochastic Gradient Descent'
    # Modified huber enables proba
    sgd_clf = SGDClassifier(loss='modified_huber', penalty='l2', shuffle=True)
    sgd_clf.fit = sgd_clf.fit(predictor, classes)

    # K Nearest Neighbor
    print '\t\tK Nearest Neighbor'
    kn_clf = KNeighborsClassifier(algorithm='ball_tree', n_neighbors=25)
    kn_clf.fit = kn_clf.fit(predictor, classes)

    print '\tTraining time: %f seconds' % (time.time() - start)
    start = time.time()

    print 'Step 4) Test set'
    tests = [x for x in gnd_test.find({},
                                      {'_id': 1,
                                       'ind': 1,
                                       'gender': 1})]

    test_predictor = [x['ind'] for x in tests]

    # Probabilities
    rf_prob = rf_clf.predict_proba(test_predictor).tolist()
    kn_prob = kn_clf.predict_proba(test_predictor).tolist()
    sgd_prob = sgd_clf.predict_proba(test_predictor).tolist()

    # Predictions
    rf_cls = rf_clf.predict(test_predictor).tolist()
    kn_cls = kn_clf.predict(test_predictor).tolist()
    sgd_cls = sgd_clf.predict(test_predictor).tolist()

    for i in range(0, len(tests)):
        # Progress counter
        if i % 10000 == 0:
            print '\t%i of %i' % (i, len(tests))

        # Method Averaging using probabilities

        # ma_prob = [round(numpy.mean([rf_prob[i][0], kn_prob[i][0]]), 2),
        #            round(numpy.mean([rf_prob[i][1], kn_prob[i][1]]), 2)]

        ma_prob = [round(numpy.mean([rf_prob[i][0], kn_prob[i][0], sgd_prob[i][0]]), 2),
                   round(numpy.mean([rf_prob[i][1], kn_prob[i][1], sgd_prob[i][1]]), 2)]
        if abs(ma_prob[0] - ma_prob[1]) > .00:
            if ma_prob[0] > ma_prob[1]:
                ma_cls = 0
            elif ma_prob[0] < ma_prob[1]:
                ma_cls = 1
        else:
            ma_cls = 9

        gnd_test.update({'_id': tests[i]['_id']},
                        {'$set': {'rf_prob': rf_prob[i],
                                  'rf_cls': rf_cls[i],
                                  'sgd_prob': sgd_prob[i],
                                  'sgd_cls': sgd_cls[i],
                                  'kn_prob': kn_prob[i],
                                  'kn_cls': kn_cls[i],
                                  'ma_prob': ma_prob,
                                  'ma_cls': ma_cls}},
                        upsert=True, multi=False)

    print '\tClassification time: %f seconds' % (time.time() - start)


def gradeMe():
    print 'Step 5) Grading:'

    # Random Forest
    total_recs = float(gnd_test.count())
    rf_correct_f = gnd_test.find({'rf_cls': 0, 'gender': 0}).count()
    rf_correct_m = gnd_test.find({'rf_cls': 1, 'gender': 1}).count()
    rf_incorrect_f = gnd_test.find({'rf_cls': 0, 'gender': 1}).count()
    rf_incorrect_m = gnd_test.find({'rf_cls': 1, 'gender': 0}).count()
    rf_correct = rf_correct_f + rf_correct_m
    rf_incorrect = rf_incorrect_f + rf_incorrect_m
    rf_pct_correct = round(float(rf_correct) / total_recs * 100, 2)
    rf_pct_incorrect = round(float(rf_incorrect) / total_recs * 100, 2)
    print '\tRandom Forest:\n\t\t%f correct\n\t\t%f incorrect' % (rf_pct_correct, rf_pct_incorrect)

    # K Nearest Neighbor
    total_recs = float(gnd_test.count())
    kn_correct_f = gnd_test.find({'kn_cls': 0, 'gender': 0}).count()
    kn_correct_m = gnd_test.find({'kn_cls': 1, 'gender': 1}).count()
    kn_incorrect_f = gnd_test.find({'kn_cls': 0, 'gender': 1}).count()
    kn_incorrect_m = gnd_test.find({'kn_cls': 1, 'gender': 0}).count()
    kn_correct = kn_correct_f + kn_correct_m
    kn_incorrect = kn_incorrect_f + kn_incorrect_m
    kn_pct_correct = round(float(kn_correct) / total_recs * 100, 2)
    kn_pct_incorrect = round(float(kn_incorrect) / total_recs * 100, 2)
    print '\tK Nearest Neighbor:\n\t\t%f correct\n\t\t%f incorrect' % (kn_pct_correct, kn_pct_incorrect)

    # Stochastic Gradient Descent
    total_recs = float(gnd_test.count())
    sgd_correct_f = gnd_test.find({'sgd_cls': 0, 'gender': 0}).count()
    sgd_correct_m = gnd_test.find({'sgd_cls': 1, 'gender': 1}).count()
    sgd_incorrect_f = gnd_test.find({'sgd_cls': 0, 'gender': 1}).count()
    sgd_incorrect_m = gnd_test.find({'sgd_cls': 1, 'gender': 0}).count()
    sgd_correct = sgd_correct_f + sgd_correct_m
    sgd_incorrect = sgd_incorrect_f + sgd_incorrect_m
    sgd_pct_correct = round(float(sgd_correct) / total_recs * 100, 2)
    sgd_pct_incorrect = round(float(sgd_incorrect) / total_recs * 100, 2)
    print '\tStochastic Gradient Descent:\n\t\t%f correct\n\t\t%f incorrect' % (sgd_pct_correct, sgd_pct_incorrect)

    # Method Averaged Result
    total_recs = float(gnd_test.count())
    ma_correct_f = gnd_test.find({'ma_cls': 0, 'gender': 0}).count()
    ma_correct_m = gnd_test.find({'ma_cls': 1, 'gender': 1}).count()
    ma_incorrect_f = gnd_test.find({'ma_cls': 0, 'gender': 1}).count()
    ma_incorrect_m = gnd_test.find({'ma_cls': 1, 'gender': 0}).count()
    ma_correct = ma_correct_f + ma_correct_m
    ma_incorrect = ma_incorrect_f + ma_incorrect_m
    ma_pct_correct = round(float(ma_correct) / total_recs * 100, 2)
    ma_pct_incorrect = round(float(ma_incorrect) / total_recs * 100, 2)
    print '\tMethod Averaged Result:\n\t\t%f correct\n\t\t%f incorrect' % (ma_pct_correct, ma_pct_incorrect)


def main():
    print '\nStarting Inference Modeling'
    print '==========================='
    prep_data(sample=100000)
    classifiers()
    gradeMe()


if __name__ == '__main__':
    main()
