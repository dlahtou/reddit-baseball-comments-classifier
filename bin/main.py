import json
import os
import pandas as pd
import pickle as pkl

from nltk.tokenize import RegexpTokenizer
from nltk.stem.wordnet import WordNetLemmatizer
from nltk.corpus import stopwords
from sklearn import naive_bayes
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics import accuracy_score
from sklearn.model_selection import train_test_split
from imblearn.under_sampling import RandomUnderSampler

import transformers

def main():
    baseball_obs, subreddit_obs = extract()

    baseball_obs = transform(baseball_obs, subreddit_obs)

    model, baseball_obs = train(baseball_obs)

    load(model, baseball_obs)


def extract():
    """
    Loads pre-downloaded files that were created using baseball_subreddit.py
        and team_subreddits.py
    """

    # load 250,000 comments from r/baseball
    basedata = []
    for i in range(0,50):
        with open(f'data/baseball/data{i}.json', 'r') as open_file:
            basedata.extend(json.load(open_file))

    # load 10,000 comments from team subreddits (300,000 total) !!large subreddits underrepresented!!
    subdata = []
    for directory in os.listdir('data'):
        for i in range(2):
            with open(f'data/{directory}/data{i}.json', 'r') as open_file:
                subdata.extend(json.load(open_file))

    return basedata, subdata

def transform(baseball_obs, subreddit_obs):
    # build dataframe from list of json objects
    comments = []
    for comment in baseball_obs:
        atts = transformers.build_row(comment)
        comments.append(atts)

    baseball_df = pd.DataFrame(comments, columns=['body', 'author', 'team', 'subreddit', 'score', 'created_utc'])

    # filter unusable data
    baseball_df = transformers.discard_removed(baseball_df)
    baseball_df = transformers.discard_non_mlb(baseball_df)
    # filter comments which are entirely quoted text
    baseball_df['body'] = baseball_df.apply(lambda row: transformers.filter_quotes(row['body']), axis=1)
    
    # perform various regex substitutions
    baseball_df = transformers.standardize_text(baseball_df, 'body')

    # tokenization
    tokenizer = RegexpTokenizer(r'\w+')
    baseball_df['tokens'] = baseball_df['body'].apply(tokenizer.tokenize)

    # aggregate dataframe by author
    baseball_df = baseball_df.groupby(['author', 'team'])['tokens'].aggregate(sum)
    baseball_df = baseball_df.reset_index()
    baseball_df.set_index('author', inplace=True)
    baseball_df = baseball_df[~baseball_df.index.duplicated(keep='first')]

    # build and transform the subreddit data, similar to above steps
    comments = []
    for comment in subreddit_obs:
        atts = transformers.build_subreddit_row(comment)
        comments.append(atts)

    subreddit_df = pd.DataFrame(comments, columns=['body', 'author'])
    subreddit_df = transformers.discard_removed(subreddit_df)

    subreddit_df['body'] = subreddit_df.apply(lambda row: transformers.filter_quotes(row['body']), axis=1)
    subreddit_df = subreddit_df[subreddit_df['author'] != 'AutoModerator']
    subreddit_df = transformers.standardize_text(subreddit_df, 'body')
    subreddit_df['tokens'] = subreddit_df['body'].apply(tokenizer.tokenize)
    subreddit_df = subreddit_df.groupby('author')['tokens'].aggregate(sum)
    subreddit_df = subreddit_df.reset_index()
    subreddit_df.set_index('author', inplace=True)
    baseball_df_authors=set(baseball_df.index)

    # add tokens from subreddit_df to corresponding author tokens in baseball_df
    for author in subreddit_df.index:
        if author in baseball_df_authors:
            baseball_df.loc[author,'tokens'].extend(subreddit_df.loc[author, 'tokens'])

    # lemmatization
    lemm = WordNetLemmatizer()
    baseball_df['tokens'] = baseball_df['tokens'].apply(lambda tokens: [lemm.lemmatize(token) for token in tokens])

    # filter stopwords
    s_words = set(stopwords.words('english'))
    baseball_stopwords = {'game', 'team', 'baseball', 'player', 'like', 'year', 'one', 'would', 'think', 'good', 'season', 'make', 'even', 'inning', 'guy', 'hit', 'really', 'still',
                          'play', 'got', '1', '2', '3', 'better', 'right', 'see', 'run', 'win', 'pitcher', 'first', 'back', 'know', 'fan', 'go',
                          'people', 'well', 'going', 'say', 'way', 'much', 'ball', 'best', 'thing', 'shit', 'around',
                          'career', 'era', 'anything', 'dude', 'hard', 'hitting', '8', 'new', '9', 'average', 'either', 'definitely',
                          'playing', 'read', 'wa', 'mlb', 'pitching', 'part', 'wrong', 'trying', 'sport', 'fun', 'home',
                          'name', 'work', 'away', 'went', 'reason', 'god', 'number', 'old', 'post', 'week', 'division', 'hate',
                          'record', 'night', 'played', 'already', 'seems', 'anyone', 'nothing', 'tonight', 'get', 'pitch', 'ha',
                          'last', 'could', 'bad', 'day', 'time', 'let', 'though', 'sure', 'probably', 'yeah', 'lol', 'also', '5', '4',
                          'two', 'lot', 'man', 'u'}
    s_words = s_words.union(baseball_stopwords)
    baseball_df['tokens'] = baseball_df.apply(lambda row: [token for token in row['tokens'] if token not in s_words], axis=1)
    
    # filter infrequent words (<10 occurrences across all data)
    freq = pd.Series(' '.join(baseball_df['token_strings']).split()).value_counts()
    inf_words = set(freq[freq <= 10].keys())
    baseball_df['tokens'] = baseball_df.apply(lambda row: [token for token in row['tokens'] if token not in inf_words], axis=1)
    baseball_df['token_strings'] = baseball_df['tokens'].apply(lambda x: ' '.join(x))

    # drop rows with two or fewer tokens
    baseball_df = baseball_df[baseball_df['tokens'].map(len) > 2]

    return baseball_df

def train(baseball_df):
    # TF-IDF vectorization
    vect = TfidfVectorizer(strip_accents='unicode', ngram_range=(1, 2))
    vectorized_comments = vect.fit_transform(baseball_df['token_strings'])

    # assign target variable
    baseball_df.reset_index(inplace=True)
    labels = baseball_df['team']

    # Train-test split (75:25)
    X_train, X_test, y_train, y_test = train_test_split(vectorized_comments, labels, random_state=10)
    
    # balance training set
    # undersample authors that do not belong to the minority class
    rus = RandomUnderSampler(random_state=10)
    X_train, y_train = rus.fit_sample(X_train, y_train)

    # run model
    NBmodel = naive_bayes.MultinomialNB()
    NBmodel.fit(X_train, y_train)
    y_pred = NBmodel.predict(X_test)
    acc = accuracy_score(y_test, y_pred)

    print(f'Accuracy Score: {acc}')

    return NBmodel, baseball_df

def load(model, data):
    with open('model.pkl', 'wb') as open_file:
        pkl.dump(model, open_file)
    
    with open('data.pkl', 'wb') as open_file:
        pkl.dump(data, open_file)

if __name__ == '__main__':
    main()
