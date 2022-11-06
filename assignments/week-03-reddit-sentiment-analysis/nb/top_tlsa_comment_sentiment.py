
import secrets_reddit
import random

from typing import Dict, List

from praw import Reddit
from praw.models.reddit.subreddit import Subreddit
from praw.models import MoreComments

from transformers import pipeline


def get_subreddit(display_name:str) -> Subreddit:
    """Get subreddit object from display name

    Args:
        display_name (str): [description]

    Returns:
        Subreddit: [description]
    """
    reddit = Reddit( # Instantiate a reddit object
        client_id=secrets_reddit.REDDIT_API_CLIENT_ID,        
        client_secret=secrets_reddit.REDDIT_API_CLIENT_SECRET,
        user_agent=secrets_reddit.REDDIT_API_USER_AGENT
        )
    
    subreddit = reddit.subreddit(display_name) # use reddit object to instantiate subreddit object
    return subreddit

def get_comments(subreddit:Subreddit, limit:int=3) -> List[str]:
    """ Get comments from subreddit

    Args:
        subreddit (Subreddit): [description]
        limit (int, optional): [description]. Defaults to 3.

    Returns:
        List[str]: List of comments
    """
    top_comments = [] # list of top-level comments
    # Loop over the 'limit' submissions for the chosen subreddit and grab all top level comments
    for submission in subreddit.top(limit=limit):
        # Loop over each comment in the specific submission (uses a 'CommentForest' object)
        for top_level_comment in submission.comments:
            # If the comment is a NOT a top level comment, then ignore it (avoids a potential error when you call comment.body)
            if isinstance(top_level_comment, MoreComments):
                continue
            # If the comment is a top level comment, append it to our stored comments for the subreddit
            top_comments.append(top_level_comment.body)
    return top_comments

def run_sentiment_analysis(comment:str) -> Dict:
    """Run sentiment analysis on comment using default distilbert model
    
    Args:
        comment (str): [description]
        
    Returns:
        str: Sentiment analysis result
    """
    sentiment_model = pipeline("sentiment-analysis") # sentiment analysis model/task
    sentiment = sentiment_model(comment) # predict sentiment of comment using huggingFace sentiment analysis task
    return sentiment[0]


if __name__ == '__main__':
    subreddit = get_subreddit("TSLA")
    comments = get_comments(subreddit)
    random.seed(1) # set seed to give reproducability of comment chosen randomly
    comment = random.choice(comments)
    sentiment = run_sentiment_analysis(comment)
    
    print(f'The comment: {comment}')
    print(f'Predicted Label is {sentiment["label"]} and the score is {sentiment["score"]:.3f}')
