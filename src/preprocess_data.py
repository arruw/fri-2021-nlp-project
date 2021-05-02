import pandas as pd
from nltk.corpus import stopwords
from nltk.stem.porter import *
# from twitter_keys import *
# from twython import Twython, TwythonError


def preprocess(text_string):
    stemmer = PorterStemmer()
    space_pattern = '\s+'
    giant_url_regex = ('http[s]?://(?:[a-zA-Z]|[0-9]|[$-_@.&+]|'
                       '[!*\(\),]|(?:%[0-9a-fA-F][0-9a-fA-F]))+')
    mention_regex = '@[\w\-]+'
    retweet_regex = '^[! ]*RT'
    parsed_text = re.sub(space_pattern, ' ', text_string)
    parsed_text = re.sub(giant_url_regex, '', parsed_text)
    parsed_text = re.sub(mention_regex, '', parsed_text)
    parsed_text = re.sub(retweet_regex, '', parsed_text)
    stemmed_words = [stemmer.stem(word) for word in parsed_text.split()]
    parsed_text = ' '.join(stemmed_words)
    return parsed_text


def cleanTweets(text):
    text = re.sub('@[A-Za-z0–9]+', '', text)  # Removing @mentions
    text = re.sub('#', '', text)  # Removing '#'
    text = re.sub('!', '', text)  # Removing '!'
    text = re.sub('\"', '', text)  # Removing '\"'
    text = re.sub(':', '', text)  # Removing ':'
    text = re.sub('RT[\s]+', '', text)  # Removing RT
    text = re.sub('https?:\/\/\S+', '', text)  # Removing hyperlink

    url = re.compile(r"https?://\S+|www\.\S+")
    text = url.sub(r"", text)  # Removing URls

    stop = set(stopwords.words("english"))
    text = [word.lower() for word in text.split() if word.lower not in stop]  # Removing stopwords

    return " ".join(text)


def preprocess_dataset6():
    df = pd.read_csv(r'../data/non-processed/dataset6/Twitter Sentiments.csv')
    df['clean_tweets'] = df['tweet'].apply(cleanTweets)
    df.to_csv('../data/processed/dataset6/twitter-sentiments.csv')


# global twitter
# twitter = Twython(apikey, apisecretkey, accesstoken, accesstokensecret)
#
#
# def get_tweet(id):
#     try:
#         temp = twitter.lookup_status(id=id)
#     except TwythonError as e:
#         print("TwythonError: {0}".format(e))
#     else:
#         tweet = dict()
#         for i in temp:
#             tweet[str(i["id"])] = i["text"]
#         return tweet


def process_dataset2():
    path = '../data/non-processed/dataset2/NAACL_SRW_2016.csv'
    df = pd.read_csv(path, index_col=False)
    tweets_id = df['id']
    labels = df['label']
    # print(get_tweet(int(tweets_id[0])))
    tweets = []
    for id in tweets_id:
        d = get_tweet(int(id))
        for key in d:
            tweets.append(d[key])

    num_labels = []
    for label in labels:
        if label == "rasism" or label == "sexism":
            num_labels.append(1)
        elif label == "none":
            num_labels.append(0)

    df['tweet'] = tweets
    df['num_labels'] = num_labels
    df.to_csv('../data/processed/dataset2/naacl_srw_2016.csv')


def preprocess_dataset1():
    df = pd.read_csv('../data/non-processed/dataset1/labeled_data.csv')
    df['clean_tweets'] = df['tweet'].apply(cleanTweets)
    tweets = []
    for tweet in df['clean_tweets']:
        tweets.append(preprocess(tweet))
    df['clean_tweets'] = tweets
    label = df['class']
    temp = []
    for l in label:
        if l == 1:
            temp.append(2)  # hate speech
        elif l == 2:
            temp.append(3)  # offensive language
        elif l == 0:
            temp.append(0)  # none
    df['labels'] = temp
    df.to_csv('../data/processed/dataset1/data.csv')


def preprocess_hasoc():
    df = pd.read_csv('../data/non-processed/english_dataset_hasoc/english_dataset.tsv', sep='\t')

    df['clean_tweets'] = df['text'].apply(cleanTweets)
    df['clean_tweets'] = df['text'].apply(preprocess)

    labels = []
    for label in df['task_2']:
        if label == 'NONE':
            labels.append(0)
        elif label == 'HATE':  # hate speech
            labels.append(2)
        elif label == 'PRFN':
            labels.append(4)  # prfn language
        elif label == 'OFFN':
            labels.append(3)  # offensive language

    df['label'] = labels

    df.to_csv('../data/processed/hasoc/en_train.csv')

    df = pd.read_csv('../data/non-processed/english_dataset_hasoc/hasoc2019_en_test-2919.tsv', sep='\t')

    df['clean_tweets'] = df['text'].apply(cleanTweets)
    df['clean_tweets'] = df['text'].apply(preprocess)

    labels = []
    for label in df['task_2']:
        if label == 'NONE':
            labels.append(0)
        elif label == 'HATE':  # hate speech
            labels.append(2)
        elif label == 'PRFN':
            labels.append(4)  # profanity language
        elif label == 'OFFN':
            labels.append(3)  # offensive language

    df['label'] = labels

    df.to_csv('../data/processed/hasoc/en_test.csv')


def preprocess_conan():
    data = pd.read_json('../data/non-processed/dataset4/CONAN.json')

    df = list(data['conan'].items())  #[0][1]['hateSpeech'])
    text = []
    classes = []
    subclass = []
    labelType = []
    labelsSubType = [set()]
    for item in df:
        text.append(item[1]['hateSpeech'])
        classes.append(item[1]['hsType'])
        subclass.append(item[1]['hsSubType'])

    data = pd.DataFrame()
    data['text'] = text

    for c in classes:
        if c == 'Islamophobia':
            labelType.append(5)
        else:
            labelType.append(0)
    # for c in subclass:
    #     if c == 'culture':
    #         labelsSubType.append(5)
    #     elif c == 'crimes':
    #         labelsSubType.append(6)
    #     elif c == 'women':
    #         labelsSubType.append(7)
    #     elif c == 'terrorism':
    #         labelsSubType.append(8)
    #     elif c == 'rapism':
    #         labelsSubType.append(9)
    #     elif c == 'islamization':
    #         labelsSubType.append(10)
    #     elif c == 'economics':
    #         labelsSubType.append(11)

    data['labelType'] = labelType
    # data['labelSubType'] = labelsSubType

    data.to_csv('../data/processed/dataset4/conan.csv', index=False)


def combine_datasets():
    df1 = pd.read_csv('../data/processed/dataset1/data.csv')
    df4 = pd.read_csv('../data/processed/dataset4/conan.csv')
    df6 = pd.read_csv('../data/processed/dataset6/twitter-sentiments.csv')
    df_hasoc1 = pd.read_csv('../data/processed/hasoc/en_train.csv')
    df_hasoc2 = pd.read_csv('../data/processed/hasoc/en_test.csv')

    df = pd.DataFrame()
    df['text'] = list(df1['clean_tweets']) + list(df4['text']) + list(df6['clean_tweets']) + \
                 list(df_hasoc1['clean_tweets']) + list(df_hasoc2['clean_tweets'])

    df['label'] = list(df1['labels']) + list(df4['labelType']) + list(df6['label']) + \
                 list(df_hasoc1['label']) + list(df_hasoc2['label'])

    df.to_csv('../data/processed/combined_data.csv', index=False)






















