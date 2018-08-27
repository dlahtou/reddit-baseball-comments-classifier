import pickle as pkl
import re


def standardize_text(df, text_field):
    df[text_field] = df[text_field].str.replace(r"http\S+", "")
    df[text_field] = df[text_field].str.replace(r"http", "")
    df[text_field] = df[text_field].str.replace(r"@\S+", "")
    df[text_field] = df[text_field].str.replace(r"[^A-Za-z0-9(),!?@\'\`\"\_\n]", " ")
    df[text_field] = df[text_field].str.replace(r"@", "at")
    df[text_field] = df[text_field].str.lower()
    return df

def filter_quotes(comment_text):
    if re.match('&gt;', comment_text) and not re.search('\n\n', comment_text):
        return ''
    return re.sub(r'&gt;.*?\n\n', '', comment_text)

def discard_removed(df):
    df2 = df[df['body'] != '[removed]'].copy()
    return df2

def discard_non_mlb(df):
    with open('pkl/teams_set.pkl', 'rb') as open_file:
        teams = pkl.load(open_file)

    df2 = df[df['team'].isin(teams)].copy()
    return df2


def build_row(comment):
    body = comment['body'].lower()
    author = comment['author']
    team = comment['author_flair_richtext'][0]['t'].lower().strip() if comment['author_flair_text'] else None
    if team == 'cincinnati redlegs':
        team = 'cincinnati reds'
    if team == 'st louis cardinals':
        team = 'st. louis cardinals'
    if team == 'california angels':
        team = 'los angeles angels'
    if team == 'florida marlins':
        team = 'miami marlins'
    elif team == 'san fransico giants' or team == 'san fransisco giants':
        team = 'san francisco giants'
    subreddit = comment['subreddit']
    score = comment['score']
    created_utc = comment['created_utc']
    
    return [body, author, team, subreddit, score, created_utc]

def build_subreddit_row(comment):
    body = comment['body'].lower()
    author = comment['author']
    
    return [body, author]