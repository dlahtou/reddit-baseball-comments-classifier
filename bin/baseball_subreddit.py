import datetime
import json

from psaw import PushshiftAPI

def main():
    '''
    Downloads r/baseball commment data for the time period from start to end (timestamps)
    
    Retrieves comments posted between the dates March 28, 2018 and August 20, 2018.
    '''

    api = PushshiftAPI()
    start = 1522281600
    print(f'finding comments going back to {datetime.datetime.fromtimestamp(start)}')
    end = 1534802400
    total_comments = 0

    # gather and save data in 5000-comment chunks
    counter = 0
    while end > start:
        print(f'searching for comments before {datetime.datetime.fromtimestamp(end)}')
        gen = api.search_comments(subreddit='baseball', limit=5000, after=start, before=end)

        comments = [comment.d_ for comment in gen]

        total_comments += len(comments)
        print(f'length {len(comments)}')
        if not comments:
            break

        with open(f'data/baseball/data{counter}.json', 'w') as open_file:
            open_file.write(json.dumps(comments))

        # assign timestamp of oldest retrieved comment as end
        end = comments[-1]['created_utc']
        counter += 1

    print(f'{total_comments} found')

if __name__ == '__main__':
    main()
