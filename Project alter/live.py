import tweepy 

import json

#consumer key , consumer secret , access token , access secret 

ckey = "G8clLXjiowIXnmBKVjOQK4Kgy"
csecret = "QO7tIFuBtM8IZmxdOqMQHaau9dvUpd8mjcEAX4tI0LsuyEasMh"
atoken = "1108737913729310721-rDpcBcoG3eCrHWXCcuiKlYCXZUc0wy"
asecret = "TTW8FFaiHQ3HWK5iI8VA7cV1xufVtJFhJ3WxuHbrxdhgV"

class listener(StreamListener):
    def on_data(self , data):
        all_data = json.loads(data)

        tweet = all_data["text"]

        print((tweet))

        return True
    
    def on_error(self , status):
        print(status)



