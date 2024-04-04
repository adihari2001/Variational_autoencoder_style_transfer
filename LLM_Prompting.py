import pandas as pd
import backoff
import tqdm
import openai

# insert api_key and organization
openai.api_key = ""
openai.organization = ""

data_path = "/work/pi_adrozdov_umass_edu/nananthamuru_umass_edu/temp/"

pn_prompt = f"""
Convert the following POSITIVE sentence to a NEGATIVE sentiment:
{{}}
"""

np_prompt = f"""
Convert the following NEGATIVE sentence to a POSITIVE sentiment:
{{}}
"""

system_prompt = f"""
You are a system that converts sentences from one sentiment to another.
You should convert from POSITIVE sentiment to NEGATIVE sentiment and vice-versa.
If asked to convert from POSITIVE to POSITIVE or from NEGATIVE to NEGATIVE, do not change the sentence.
You only respond with the lowercase sentence without any punctuation.

"""

examples_prompt = f"""
Here are a few examples of sentences in both POSITIVE and NEGATIVE sentiments:
POSITIVE: it is a good toasted hoagie
NEGATIVE: it is a bad toasted hoagie

POSITIVE: fast and friendly service
NEGATIVE: slow and rude service

POSITIVE: i love everything about this place
NEGATIVE: i hate everything about this place

POSITIVE: it is a great hometown neighborhood bar with good people and friendly staff
NEGATIVE: it is a terrible hometown neighborhood bar with bad people and unfriendly staff

"""


@backoff.on_exception(backoff.expo, openai.error.RateLimitError)
def get_llm_response(sent, target=0, zero_shot=False):
    sys_p = system_prompt
    if not zero_shot:
        sys_p += examples_prompt
    # llm_response = openai.ChatCompletion.create(
    #     model="text-davinci-003",
    #     prompt=examples_prompt + (p_prompt if target == 0 else n_prompt).format(sent),
    #     max_tokens=256,
    #     n=1
    # )['choices'][0]
    llm_response = openai.ChatCompletion.create(
        model="gpt-3.5-turbo",
        messages=[
            {"role": "system", "content": sys_p},
            {"role": "user", "content": (np_prompt if target == 0 else pn_prompt).format(sent)}
        ],
        max_tokens=256,
        n=1
    )['choices'][0].message
    return llm_response['content'].split('\n')[-1].lower()


with open(data_path + 'yelp_sampled_data.txt') as f:
    X = f.readlines()

with open(data_path + 'yelp_sampled_labels.txt') as f:
    Y = f.readlines()

print("Total Data Points:", len(Y))

df = pd.DataFrame()
for idx, (input, label) in tqdm.tqdm(enumerate(zip(X, Y))):
    target = 1 if label.strip() == "pos" else 0
    pred = get_llm_response(input.strip(), target=target, zero_shot=True)
    add_df = pd.DataFrame([{'input': input.strip(), 'label': label.strip(), 'pred': pred.strip()}])
    df = pd.concat([df, add_df])

df.to_csv(data_path + 'llm_zero_shot_outputs.csv', index=False)
