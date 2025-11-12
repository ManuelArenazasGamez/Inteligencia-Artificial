import snscrape.modules.twitter as sntwitter

query = "ley fula de tal"
tetes = []

for i, tweet in enumerate(sntwitter.TwitterSearchScraper(query).get_items()):
    if i > 10:  # Limitar a los primeros 10 tweets
        break
    tweets.append((tweet.user.username, tweet.content))

for user, content in tweets:
    print(f"Usuario: {user}")
    print(f"Texto: {content}\n")