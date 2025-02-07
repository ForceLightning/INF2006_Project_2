"""This module contains the code for the sentiment analysis task for the Twitter Airline dataset.
"""

from typing import Literal
import nltk
from nltk.corpus import wordnet as wn
from nltk.corpus import sentiwordnet as swn
from nltk.tag import pos_tag
from nltk.tokenize import word_tokenize
from nltk.stem import WordNetLemmatizer
from vaderSentiment.vaderSentiment import SentimentIntensityAnalyzer

__author__ = "Cheng Yi Xing"


class SentimentAnalysis:
    """
    A class for performing sentiment analysis on text.

    :ivar lemmatizer: A lemmatizer object for lemmatizing words.
    :vartype lemmatizer: WordNetLemmatizer
    :ivar sid_obj: A sentiment intensity analyzer object for VADER sentiment analysis.
    :vartype sid_obj: SentimentIntensityAnalyzer
    :ivar threshold: The threshold value for determining positive, negative, or neutral sentiment.
    :vartype threshold: float
    :ivar increased_threshold: The increased threshold value for determining
        positive or negative sentiment.
    :vartype increased_threshold: float
    :ivar decreased_threshold: The decreased threshold value for determining
        positive or negative sentiment.
    :vartype decreased_threshold: float
    :ivar stop_words: A set of stop words for removing from the text.
    :vartype stop_words: set
    :ivar negations: A set of negation words for handling negation in sentiment analysis.
    :vartype negations: set
    """

    def __init__(self):
        self.lemmatizer = WordNetLemmatizer()
        self.sid_obj = SentimentIntensityAnalyzer()
        self.threshold = 0.1
        self.increased_threshold = 0.2
        self.decreased_threshold = 0.05
        self.stop_words = set(nltk.corpus.stopwords.words("english"))
        self.negations = {
            "not",
            "never",
            "no",
            "nobody",
            "none",
            "nor",
            "nothing",
            "nowhere",
        }

    def get_wordnet_pos(self, treebank_tag: str) -> str | None:
        """Maps the treebank part-of-speech tags to WordNet part-of-speech tags.

        :param treebank_tag: The treebank part-of-speech tag.
        :type treebank_tag: str
        :return: The corresponding WordNet part-of-speech tag, or None if no mapping is found.
        :rtype: str or None
        """
        if treebank_tag.startswith("J"):
            return wn.ADJ
        elif treebank_tag.startswith("V"):
            return wn.VERB
        elif treebank_tag.startswith("N"):
            return wn.NOUN
        elif treebank_tag.startswith("R"):
            return wn.ADV
        else:
            return None

    def get_sentiment_vader(self, sentence: str) -> str | None:
        """Get the sentiment of a sentence using VADER sentiment analysis.

        :param sentence: The sentence to analyze.
        :type sentence: str
        :return: The sentiment of the sentence. Possible values are
            "positive", "negative", or "neutral".
        :rtype: str | None
        """
        if not sentence:
            return None
        sentiment_dict = self.sid_obj.polarity_scores(sentence)

        if sentiment_dict["compound"] >= 0.05:
            return "positive"
        elif sentiment_dict["compound"] <= -0.05:
            return "negative"
        else:
            return "neutral"

    def get_sentiment(
        self, sentence: str
    ) -> Literal["positive", "negative", "neutral"] | None:
        """Analyzes the sentiment of a given sentence.

        :param sentence: The input sentence to analyze.
        :type sentence: str
        :return: The sentiment of the sentence, which can be "positive", "negative", or "neutral".
        :rtype: str
        """
        if not sentence:
            return None

        tokenized_text = word_tokenize(sentence)
        tagged_text = pos_tag(tokenized_text)

        pos_score = 0
        neg_score = 0
        token_count = 0

        for word, tag in tagged_text:
            wn_tag = self.get_wordnet_pos(tag)
            if wn_tag not in (wn.NOUN, wn.ADJ, wn.ADV):
                continue

            lemma = wn.lemmas(word, pos=wn_tag)
            if not lemma:
                continue

            lemma = lemma[0]
            synset = lemma.synset()
            swn_synset = swn.senti_synset(synset.name())

            pos_score += swn_synset.pos_score()
            neg_score += swn_synset.neg_score()
            token_count += 1

        # Normalize the score to be between -1 and 1
        if token_count:
            normalized_score = (pos_score - neg_score) / token_count
        else:
            normalized_score = 0  # Neutral if no tokens are processed

        # Return negative, neutral or positive based on the normalized score
        if normalized_score >= self.threshold:
            return "positive"
        elif normalized_score <= -self.threshold:
            return "negative"
        else:
            return "neutral"

    def get_sentiment_stop_words_removed(
        self, sentence: str
    ) -> Literal["positive", "negative", "neutral"] | None:
        """Calculates the sentiment of a given sentence after removing stop words.

        :param sentence: The input sentence to analyze.
        :type sentence: str
        :return: The sentiment of the sentence, which can be "positive", "negative", or "neutral".
        :rtype: str | None
        """
        if not sentence:
            return None

        tokenized_text = word_tokenize(sentence)
        tagged_text = pos_tag(tokenized_text)

        # Remove stop words
        tagged_text = [
            (word, tag)
            for word, tag in tagged_text
            if word.lower() not in self.stop_words
        ]

        pos_score = 0
        neg_score = 0
        token_count = 0

        for word, tag in tagged_text:
            wn_tag = self.get_wordnet_pos(tag)
            if wn_tag not in (wn.NOUN, wn.ADJ, wn.ADV):
                continue

            lemma = wn.lemmas(word, pos=wn_tag)
            if not lemma:
                continue

            lemma = lemma[0]
            synset = lemma.synset()
            swn_synset = swn.senti_synset(synset.name())

            pos_score += swn_synset.pos_score()
            neg_score += swn_synset.neg_score()
            token_count += 1

        # Normalize the score to be between -1 and 1
        if token_count:
            normalized_score = (pos_score - neg_score) / token_count
        else:
            normalized_score = 0

        # Return negative, neutral or positive based on the normalized score
        if normalized_score >= self.threshold:
            return "positive"
        elif normalized_score <= -self.threshold:
            return "negative"
        else:
            return "neutral"

    def get_sentiment_inverse_if_negative(
        self, sentence: str
    ) -> Literal["positive", "negative", "neutral"] | None:
        """Calculates the sentiment of a given sentence using the SentiWordNet lexicon.
            If the sentiment is negative, the scores are inverted.

        :param sentence: The input sentence to analyze.
        :type sentence: str
        :return: The sentiment of the sentence, which can be "positive", "negative", or "neutral".
        :rtype: str | None
        """
        if not sentence:
            return None

        tokenized_text = word_tokenize(sentence)
        tagged_text = pos_tag(tokenized_text)

        pos_score = 0
        neg_score = 0
        token_count = 0
        negation_present = False  # Flag to track negation

        for word, tag in tagged_text:
            if word.lower() in self.negations:
                negation_present = not negation_present  # Toggle negation flag
                continue  # Skip the negation word itself

            wn_tag = self.get_wordnet_pos(tag)
            if wn_tag not in (wn.NOUN, wn.ADJ, wn.ADV):
                continue

            lemma = wn.lemmas(word, pos=wn_tag)
            if not lemma:
                continue

            lemma = lemma[0]
            synset = lemma.synset()
            swn_synset = swn.senti_synset(synset.name())

            if negation_present:
                # Invert scores if negation is present
                pos_score += swn_synset.neg_score()
                neg_score += swn_synset.pos_score()
            else:
                pos_score += swn_synset.pos_score()
                neg_score += swn_synset.neg_score()

            token_count += 1
            negation_present = False  # Reset negation flag after processing

        # Normalize the score to be between -1 and 1
        if token_count:
            normalized_score = (pos_score - neg_score) / token_count
        else:
            normalized_score = 0

        # Return negative, neutral or positive based on the normalized score
        if normalized_score >= self.threshold:
            return "positive"
        elif normalized_score <= -self.threshold:
            return "negative"
        else:
            return "neutral"

    def get_sentiment_higher_threshold(
        self, sentence: str
    ) -> Literal["positive", "negative", "neutral"] | None:
        """Calculates the sentiment of a given sentence using a higher threshold.

        :param sentence: The input sentence to analyze.
        :type sentence: str
        :return: The sentiment of the sentence, which can be "positive", "negative", or "neutral".
        :rtype: str | None
        """
        if not sentence:
            return None

        tokenized_text = word_tokenize(sentence)
        tagged_text = pos_tag(tokenized_text)

        pos_score = 0
        neg_score = 0
        token_count = 0

        for word, tag in tagged_text:
            wn_tag = self.get_wordnet_pos(tag)
            if wn_tag not in (wn.NOUN, wn.ADJ, wn.ADV):
                continue

            lemma = wn.lemmas(word, pos=wn_tag)
            if not lemma:
                continue

            lemma = lemma[0]
            synset = lemma.synset()
            swn_synset = swn.senti_synset(synset.name())

            pos_score += swn_synset.pos_score()
            neg_score += swn_synset.neg_score()
            token_count += 1

        # Normalize the score to be between -1 and 1
        if token_count:
            normalized_score = (pos_score - neg_score) / token_count
        else:
            normalized_score = 0

        # Return negative, neutral or positive based on the normalized score
        if normalized_score >= self.increased_threshold:
            return "positive"
        elif normalized_score <= -self.increased_threshold:
            return "negative"
        else:
            return "neutral"

    def get_sentiment_lower_threshold(
        self, sentence: str
    ) -> Literal["positive", "negative", "neutral"] | None:
        """Calculates the sentiment of a given sentence based on a lower threshold.

        :param sentence: The input sentence to analyze.
        :type sentence: str
        :return: The sentiment of the sentence, which can be "positive", "negative", or "neutral".
        :rtype: str | None
        """
        if not sentence:
            return None

        tokenized_text = word_tokenize(sentence)
        tagged_text = pos_tag(tokenized_text)

        pos_score = 0
        neg_score = 0
        token_count = 0

        for word, tag in tagged_text:
            wn_tag = self.get_wordnet_pos(tag)
            if wn_tag not in (wn.NOUN, wn.ADJ, wn.ADV):
                continue

            lemma = wn.lemmas(word, pos=wn_tag)
            if not lemma:
                continue

            lemma = lemma[0]
            synset = lemma.synset()
            swn_synset = swn.senti_synset(synset.name())

            pos_score += swn_synset.pos_score()
            neg_score += swn_synset.neg_score()
            token_count += 1

        # Normalize the score to be between -1 and 1
        if token_count:
            normalized_score = (pos_score - neg_score) / token_count
        else:
            normalized_score = 0

        # Return negative, neutral or positive based on the normalized score
        if normalized_score >= self.decreased_threshold:
            return "positive"
        elif normalized_score <= -self.decreased_threshold:
            return "negative"
        else:
            return "neutral"
