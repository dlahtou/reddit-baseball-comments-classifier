import datetime
import json
import os

from psaw import PushshiftAPI

def main():
    """Fetches all reddit comments from every baseball team subreddit

    Retrieves comments posted between the dates March 28, 2018 and August 20, 2018.
    """

    api = PushshiftAPI()
    start_time = 1522281600

    subreddits = {'azdiamondbacks', 'ColoradoRockies',
                  'Dodgers', 'SFGiants', 'Padres',
                  'CHICubs', 'Brewers', 'Cardinals',
                  'Buccos', 'Reds', 'Braves',
                  'Phillies', 'Nationals', 'NewYorkMets',
                  'letsgofish', 'Astros', 'oaklandathletics', 'Mariners',
                  'AngelsBaseball', 'TexasRangers', 'WahoosTipi',
                  'MinnesotaTwins', 'MotorCityKitties', 'WhiteSox',
                  'KCRoyals', 'RedSox', 'NYYankees',
                  'TampaBayRays', 'TorontoBlueJays', 'Orioles'
                 }

    for subreddit in subreddits:
        if subreddit in os.listdir('data'):
            continue
        print(f'finding comments going back to {datetime.datetime.fromtimestamp(start_time)}')
        print(f'searching subreddit {subreddit}')
        end_time = 1534802400
        total_comments = 0

        os.mkdir(f'data/{subreddit}')

        counter = 0
        while end_time > 1522281600:
            print(f'searching for comments before {datetime.datetime.fromtimestamp(end_time)}')
            gen = api.search_comments(subreddit=subreddit, limit=5000, after=start_time, before=end_time)

            comments = [comment.d_ for comment in gen]

            total_comments += len(comments)
            print(f'length {len(comments)}')
            if not comments:
                break

            with open(f'data/{subreddit}/data{counter}.json', 'w') as open_file:
                open_file.write(json.dumps(comments))

            end_time = comments[-1]['created_utc']
            counter += 1

        print(f'{total_comments} found')


if __name__ == '__main__':
    main()
