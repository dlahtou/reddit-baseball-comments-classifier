<h1>Project Fletcher</h1>
<h2>Classifying Reddit Commenters by Their Preferred Baseball Team</h2>
Dan Ah Tou

<h2>Project Design</h2>
The goal of my project was to analyze users’ comments on the baseball-related pages of reddit, and to use this data to try and classify them as a fan of one of the 30 MLB teams. Data for this project was gathered from the baseball subreddit (r/baseball) and various MLB team subreddits (r/CHIcubs, r/NYYankees, etc.).

<h2>Tools</h2


Data
Data for this project was sourced from the reddit API using pushshift.io. Using these sources, I was able to gather data from the beginning of this year’s baseball season to the present time, which is the date range 3/28/18-8/20/18. However, I was only able to incorporate a portion of this data into a model, due to the processing constraints of my laptop. In the end, my model uses comment data from the most recent four weeks in my downloaded dataset, or the date range 7/24/18-8-20/18. This timeframe contains 550,000 reddit comments.

Reddit comments as downloaded from the API contain the full texts written by reddit users, as well as additional metadata about the comments. Metadata fields that were of particular use to me included the post date, username, and user flair. A user’s flair, in the context of r/baseball, is an icon that a user selects for themself which indicates their preferred team. For example, a user who follows the Oakland A’s will select a green and gold styled “A’s” logo to appear next to their username for all content they post on the site. I was able to use this metadata label as the target variable for my classification model.

Comment data was processed first by filtering out any unusable data -- namely, removing bot posts, posts with no team label, or posts which consisted entirely of comment text. Next, comments were grouped by their authors, tokenized, and lemmatized. After this filtering and grouping step was performed, comments had been aggregated under approximately 28,000 authors. Finally, stop words were removed. For this dataset, many generic baseball terms such as “player, game, ball” were considered stop words, and were filtered from the dataset.

Algorithms
Latent Dirichlet Allocation was used to cluster users’ comments into topics for further analysis. This was accomplished using 
I used a Naive Bayes algorithm to perform the final version of classification in this project. This algorithm had the best performance of any algorithm in the classification problem, with an accuracy of 27%. A gradient boosting model returned an accuracy of 14%, and a dummy classifier baseline model returned an accuracy of 3.3%.

What I’d Do Differently Next Time
I would implement some form of dimensionality reduction on my vectorized data to reduce overfitting of my model. 
