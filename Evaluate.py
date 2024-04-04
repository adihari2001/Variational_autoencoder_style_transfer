import math
import random

import fasttext
import dill as pickle

import nltk
from nltk.tokenize import word_tokenize
from nltk.corpus import stopwords
from nltk.sentiment.vader import SentimentIntensityAnalyzer
from nltk.util import ngrams
from nltk.lm.preprocessing import padded_everygram_pipeline, pad_both_ends
from nltk.lm import KneserNeyInterpolated

nltk.download('punkt')          # tokenizer
nltk.download('vader_lexicon')  # sentiment
nltk.download('stopwords')      # stop words

class FasttextEvaluate:
    """
    A class to evaluate the style transfer accuracy of a style transfer task.
    """

    def __init__(self, file_path=None):
        self.model = None
        if file_path:
            self.model = fasttext.load_model(file_path)

    def preprocess_data(self, text_file_path, labels_file_path, out_file_path):
        with open(text_file_path, "r") as f:
            text = f.read().split("\n")

        with open(labels_file_path, "r") as f:
            labels = f.read().split("\n")

        # preprocessing and shuffling the data
        new_data = [f"__label__{labels[i]} {text[i]}\n" for i in range(len(text))]
        random.shuffle(new_data)

        # saving the preprocessed data
        with open(out_file_path, "w") as f:
            for line in new_data:
                f.write(line)

    def train_model(self, input_file_path):
        """
        Train the fasttext classifier model.

        Args:
            input_file_path: preprocessed text file (use FasttextEvaluate.preprocess_data)

        Returns:
            None
        """
        self.model = fasttext.train_supervised(input_file_path)
    
    def test_model(self, input_file_path):
        """
        Test the fasttext classifier model.

        Args:
            input_file_path: preprocessed text file (use FasttextEvaluate.preprocess_data)

        Returns:
            accuracy
        """
        return self.model.test(input_file_path)
    
    def save_model(self, model_save_path):
        if self.model:
            self.model.save_model(model_save_path)
        else:
            print("Cannot save untrained model.")
    
    def score(self, text, label):
        """
        Calculates the confidence score of the text being classified in the label.

        Args:
            text:  style transferred text
            label: target label

        Returns:
            conf_score: (float) confidence score (range 0-1)
        """
        labels, scores = self.model.predict(text, 2)
        return scores[0] if (labels[0][9:] == label) else scores[1]
    
class WordOverlapEvaluate:
  """
  A class to evaluate the content preservation score of a style transfer task 
  (stopwords & sentiment words removed).
  """

  def __init__(self):
    self.sent_int_a = SentimentIntensityAnalyzer()
    self.stop_words = set(stopwords.words('english'))

  def score(self, text1, text2):
    """
    Calculates the word overlap rate of the given texts.
    Performs stopwords & sentiment words removal.

    Args:
        text1, text2: input/output strings

    Returns:
        word_overlap: (float) word overlap rate (range 0-1)
        
    """
    w1 = [w for w in word_tokenize(text1) if w not in self.stop_words and math.isclose(self.sent_int_a.polarity_scores(w)["compound"], 0)]
    w2 = [w for w in word_tokenize(text2) if w not in self.stop_words and math.isclose(self.sent_int_a.polarity_scores(w)["compound"], 0)]

    union = len(set(w1) | set(w2))
    if union == 0:
        return 0
    
    inter = len(set(w1) & set(w2))

    return inter / union
        
class TriGramEvaluate:
    """
    A class to evaluate the language fluency score of a style transfer task.
    """
    def __init__(self, file_path=None):
        self.model = None
        if file_path:
            with open(file_path, "rb") as f:
                self.model = pickle.load(f)

    def train_model(self, input_file_path):
        """
        Train the trigram language model.

        Args:
            input_file_path: input text file of sentences.

        Returns:
            None
        """
        with open(input_file_path, "r") as f:
            text = [word_tokenize(s.strip()) for s in f.readlines()]

        tokens, vocab = padded_everygram_pipeline(3, text)

        self.model = KneserNeyInterpolated(3)
        self.model.fit(tokens, vocab)

    def save_model(self, model_save_path):
        if self.model:
            with open(model_save_path, "wb") as f:
                pickle.dump(self.model, f)

    def score(self, text):
        """
        Calculates the perplexity of the input text with a language
        model trained on the corpus (see TriGramEvaluate.train_model)

        Args:
            text: input string

        Returns:
            perplexity: (float) perplexity of the text (range 0-inf)
            
        """
        text = word_tokenize(text)
        text = pad_both_ends(text, 3)
        text = ngrams(text, 3)
        text = list(text)

        return self.model.perplexity(text)
    
def geometric_mean(values):
    return math.prod(values) ** (1/len(values))

class Evaluate:
    """
    A wrapper class to evaluate style transfer task on 3 metrics.
        1. Style Transfer Accuracy
        2. Content Preservation
        3. Language Fluency
        [4. Geometric mean of above 3 metrics.]
    """
    def __init__(self, fasttext_model_path, trigram_model_path):
        self.ft_model = FasttextEvaluate(fasttext_model_path)
        self.wo_model = WordOverlapEvaluate()
        self.tg_model = TriGramEvaluate(trigram_model_path)

    def score(self, trans_text, trans_label, orig_text, st=True, cp=True, lf=True):
        """
            Returns the Style Transfer Accuracy (st),
            the Content Preservation (cp), and
            the Language Fluency (lf) scores. 
            (and the geometric mean of all three.)

            Args:
                trans_text : style transferred sentence
                trans_label: style transferred label
                orig_text  : original input text

            Returns:
                dictionary -> ("st", "cp", "lf", "gm")
        """

        metrics = {}
        gm_metrics = []
        if st:
            st_score = self.ft_model.score(trans_text, trans_label)
            gm_metrics.append(st_score)
            metrics["st"] = st_score
        
        if cp:
            cp_score = self.wo_model.score(trans_text, orig_text)
            gm_metrics.append(cp_score)
            metrics["cp"] = cp_score

        if lf:
            lf_score = self.tg_model.score(trans_text)
            gm_metrics.append(1/lf_score)
            metrics["lf"] = lf_score

        metrics["gm"] = geometric_mean(gm_metrics)
        
        return metrics