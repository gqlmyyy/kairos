import time
import feedparser
from data.news.fetcher import RSS_FEEDS, FOREX_KEYWORDS

def main():
    for feed_url, source_name in RSS_FEEDS:
        try:
            feed = feedparser.parse(feed_url)
            print('---', source_name, feed_url)
            print('entries:', len(feed.entries))
            for entry in feed.entries[:5]:
                headline = entry.get('title','').strip()
                summary = entry.get('summary','')[:200]
                combined = (headline+' '+summary).lower()
                matched = [kw for kw in FOREX_KEYWORDS if kw in combined]
                print('title:', headline[:90])
                print('matched:', matched[:10])
                pub_time = entry.get('published_parsed')
                # don't blow up on missing
                if pub_time:
                    pub_ts = time.mktime(pub_time)
                    hours_old = (time.time()-pub_ts)/3600
                    print('hours_old:', f'{hours_old:.2f}')
                else:
                    print('hours_old: N/A')
        except Exception as e:
            print('ERROR', source_name, e)

if __name__ == '__main__':
    main()

