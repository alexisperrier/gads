import feedparser
# http://spectrum.ieee.org/static/rss

feeds = [ 'http://spectrum.ieee.org/rss/blog/energywise/fulltext',
'http://spectrum.ieee.org/rss/blog/cars-that-think/fulltext',
'http://spectrum.ieee.org/rss/blog/the-human-os/fulltext',
'http://spectrum.ieee.org/rss/blog/riskfactor/fulltext',
'http://spectrum.ieee.org/rss/blog/nanoclast/fulltext',
'http://spectrum.ieee.org/rss/blog/tech-talk/fulltext',
'http://spectrum.ieee.org/rss/blog/view-from-the-valley/fulltext',
'http://spectrum.ieee.org/rss/aerospace/fulltext',
'http://spectrum.ieee.org/rss/at-work/fulltext',
'http://spectrum.ieee.org/rss/blog/automaton/fulltext',
'http://strata.oreilly.com/feed',
'http://feeds.arstechnica.com/arstechnica/technology-lab',
'http://feeds.arstechnica.com/arstechnica/gadgets',
'http://feeds.arstechnica.com/arstechnica/business',
'http://feeds.arstechnica.com/arstechnica/security',
'http://feeds.arstechnica.com/arstechnica/tech-policy',
'http://feeds.arstechnica.com/arstechnica/apple',
'http://feeds.arstechnica.com/arstechnica/gaming',
'http://feeds.arstechnica.com/arstechnica/science',
'http://feeds.arstechnica.com/arstechnica/multiverse',
'http://feeds.arstechnica.com/arstechnica/cars',
'http://feeds.arstechnica.com/arstechnica/staff-blogs',
]

documents = []

for rss in feeds:
    llog = feedparser.parse(rss)
    print("%d entries for %s" % (len(llog.entries), rss) )
    for entry in llog.entries:
        content = entry.content[0].value

        documents.append(content)

pickle.dump(documents, '../data/techno_feed.out')

import pickle
file = open("../data/techno_feed.txt",'rb')
docs = pickle.load(file)




