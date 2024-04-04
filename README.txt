Overall Readme for all 3 different ipynb files. You can execute the files simply by running all the cells and changing the paths to the dataset. 

VAE_Counterfactuals.ipynb
--YELP Dataset Class 
--Variational AutoEncoder Class
--Experiment Class
	-- Training
	-- Validation
	-- Testing reconstruction
	-- Saving Style Embeddings
	-- Style Transfer
	-- Style Transfer using CounterFactuals
-- Training CounterFactuals - MLP
-- Data Processing for Human Evaluation
-- Human Evaluation Aggregation and Plots

Data_Preprocessing.ipynb
-- Preprocessing
-- Vocab Creation
-- Word2Vec Model
-- BOW, Word Embeddings
-- T-SNE plots

LLM_Prompting.py
-- Prompting gpt-3.5-turbo

Evaluation.py
-- KN Smoothed Trigram Model
-- FastText Classifer
-- Word Overlap

Plots.py
-- Losses Plots

All the code was written only by the project team members.


LLM_Prompting.py:
------------------------ Imports ------------------------
!pip install openai
!pip install backoff

--------------------------Usage--------------------------
You will need an OpenAI account and an API_KEY (and 
optionally an organization as well) to run this code.

Fill in the details and then execute the file. By 
default, the code will run zero-shot prompting. You can 
set `zero_shot` param to False in the call to 
`get_llm_response` to run few-shot prompting.
---------------------------------------------------------

Evaluate.py:
------------------------ Imports ------------------------
!pip install fasttext
!pip install dill

%cd /content/drive/MyDrive/NLPPROJECT/evaluate/
from evaluate import Evaluate

------------------------- Usage -------------------------
fasttext_model_path = "/content/drive/MyDrive/NLPPROJECT/evaluate/models/yelp_trained_model.bin"
trigram_model_path = "/content/drive/MyDrive/NLPPROJECT/evaluate/models/KN_trigram_model.pkl"

em = Evaluate(fasttext_model_path, trigram_model_path)
em.score(<transferred_text>: string, <transferred_label>: string, <original_text>: string)

------------------------ Returns ------------------------
the em.score() function returns a dictionary with:
	1. "st": style transfer accuracy
	2. "cp": content preservation score
	3. "lf": language fluency perplexity score
	4. "gm": geometric mean of st, cp and 1/lf
---------------------------------------------------------