import tweepy
from textblob import TextBlob
import csv

consumer_api_key = 'jNeMXRs2lmlir4nGQ1DWU5Irk'
consumer_api_secret_key = 'LaYAXDyr0N6XeO36Y5WCVBEIMkYfFYMJ2Rkc8dNmpV992qJvxh'

access_token = '830085441332314112-cmWWI7vHbXq1B6WGxHOLWNvkpVXzvKm'
access_token_secret = 'iyK2untO3quM90fhcisZN7cSxtAepqaeiHlrlvSA8hFJq' 

auth = tweepy.OAuthHandler(consumer_api_key, consumer_api_secret_key)
auth.set_access_token(access_token, access_token_secret)

api = tweepy.API(auth)

public_tweets = api.search('Sadhguru')

csvfile = open('sentiments.csv', 'wb')
csv_writer = csv.writer(csvfile, delimiter=',')
positive = 0.5


for tweet in public_tweets:
	print(tweet.text)
	analysis = TextBlob(tweet.text)
	fact = analysis.sentiment.polarity
	opinion = analysis.sentiment.subjectivity
	label = 'positive' if ((fact >= positive and opinion < positive) or (fact == 0 and opinion == 0)) else 'negitive'
	csv_writer.writerow([tweet.text.encode("utf-8"), label.encode("utf-8"), fact, opinion])


