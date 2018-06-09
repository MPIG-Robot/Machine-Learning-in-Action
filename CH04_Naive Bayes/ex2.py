

import bayes
import feedparser

sports = feedparser.parse('rsssports.xml')
tec = feedparser.parse('rsstec.xml')

print bayes.getTopWords(sports,tec)
