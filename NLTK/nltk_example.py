# Boise Data Science Meetup
# NLTK Example
# June 2016
# Randall Shane, PhD

import nltk
from nltk import word_tokenize
from nltk.corpus import stopwords
from nltk.stem.snowball import SnowballStemmer


# NLTK Setup
tagger = nltk.UnigramTagger(nltk.corpus.brown.tagged_sents())
lemma = nltk.WordNetLemmatizer()
stemmer = SnowballStemmer("english",
                          ignore_stopwords=True)
stopwords = stopwords.words('english')

text = '''Albert Einstein was born on March 14, 1878 in Ulm, Germany. His parents Herman and Rauline Einstein were very worried about young Einstein because he was very slow to lean how to speak. When he was young he had no mark of being genius. He was the worst in class. When he was young his parents moved several times looking for a place to open businesses. His parents settled in Italy when he was 15. He soon was expelled from school in Germany and joined his family in Italy. He finished high school in Switzerland; where he graduated with a teaching degree from the Swiss Federal Institute of Technology. However he did not find a job until 1902. At the Swiss patent office, he worked there for seven years. ln 1903, he married Maria Marie. Albert Einstein conceptualized the theories of general relativity and special relativity. He came to realize that the universe was not made up of three dimensional space as was commonly accepted, but four dimensional space-time. The fourth dimension being that of time. Einstein made other great discoveries, such as the speed of light.'''
text = text.encode(encoding='utf8', errors='ignore')

# NLTK STEPS:

# tokenization
block = word_tokenize(text)
block = [x.strip(' ') for x in block]
print 'Tokenized: ', block[0:10]
print '\n'

# stemming
block = [stemmer.stem(x) for x in block]
print 'Stemms: ', block[0:10]
print '\n'

# lemmatization
block = [lemma.lemmatize(x) for x in block]
print 'Lemma: ', block[0:10]
print '\n'

# length
block = [x for x in block if len(x) > 2]
print 'Length: ', block[0:10]
print '\n'

# stopwords
block = [x for x in block if x not in stopwords]
print 'Stopwords: ', block[0:10]
print '\n'

# tagging - NN: noun, NNS: plurals, VB: verb
# specific types:
types = ['NN', 'NNS', 'VB']
block = [x[0] for x in tagger.tag(block) if x[1] in types]
tags = [x for x in tagger.tag(block)]
print 'Tags: ', tags[0:10]
print '\n'

# frequency distribution
fdist1 = nltk.FreqDist(block)

print '\nCommon Words: ', fdist1.most_common(n=10)
print '\n'
