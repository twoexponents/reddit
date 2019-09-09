from vaderSentiment.vaderSentiment import SentimentIntensityAnalyzer

analyser = SentimentIntensityAnalyzer()

def sentiment_analyzer_scores(sentence):
    score = analyser.polarity_scores(sentence)
    print ("{:-<40} {}".format(sentence, str(score)))

sentiment_analyzer_scores("hi my name is khan")
