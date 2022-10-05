from turtle import back
from transformers import MarianMTModel, MarianTokenizer
import spacy
import random
import copy
import os
import json
from pathlib import Path 
from args import get_aug_dataset_args


# Helper function to download data for a language
def download(model_name):
  tokenizer = MarianTokenizer.from_pretrained(model_name)
  model = MarianMTModel.from_pretrained(model_name)
  return tokenizer, model

# download model for English -> Romance
tmp_lang_tokenizer, tmp_lang_model = download('Helsinki-NLP/opus-mt-en-ROMANCE')
# download model for Romance -> English
src_lang_tokenizer, src_lang_model = download('Helsinki-NLP/opus-mt-ROMANCE-en')

def translate(texts, model, tokenizer, language):
  """Translate texts into a target language"""
  # Format the text as expected by the model
  formatter_fn = lambda txt: f"{txt}" if language == "en" else f">>{language}<< {txt}"
  original_texts = [formatter_fn(txt) for txt in texts]

  # Tokenize (text to tokens)
  tokens = tokenizer.prepare_seq2seq_batch(original_texts)

  # Translate
  translated = model.generate(**tokenizer(texts, return_tensors="pt", padding=True))

  # Decode (tokens to text)
  translated_texts = tokenizer.batch_decode(translated, skip_special_tokens=True)

  return translated_texts

def back_translate(texts, language_src, language_dst):
  """Implements back translation"""
  # Translate from source to target language
  translated = translate(texts, tmp_lang_model, tmp_lang_tokenizer, language_dst)

  # Translate from target language back to source language
  back_translated = translate(translated, src_lang_model, src_lang_tokenizer, language_src)

  return back_translated

def gen_back_trans(data, prob_context, prob_question, language): 
    aug_data = data.copy()
    aug_data["data"] = []
    split_sents = spacy.load('en_core_web_sm')
    for i, dict_ in enumerate(data['data']):
        context = data['data'][i]['paragraphs'][0]['context']
        context_sents = [str(i) for i in split_sents(context).sents]
        if (random.random()<prob_context):
            context_sents_trans = back_translate(context_sents, 'en', language)
            data['data'][i]['paragraphs'][0]['context'] = ' '.join(context_sents_trans)
        
        for j in range(len(data['data'][i]['paragraphs'][0]['qas'])):
            question = data['data'][i]['paragraphs'][0]['qas'][j]['question']
            question_sents = [str(i) for i in split_sents(question).sents]
            if (random.random()<prob_question):
                question_sents_trans = back_translate(question_sents, 'en', language)
                data['data'][i]['paragraphs'][0]['qas'][j]['question'] = ' '.join(question_sents_trans)
        
        aug_data['data'].append(copy.deepcopy(data['data'][i]))

    random.shuffle(aug_data["data"])
    return aug_data

def main():
  args = get_aug_dataset_args()
  if not os.path.exists(args.save_dir):
    os.makedirs(args.save_dir)
  dataset_dict = None

  with open(f"{args.datasets_dir}/{args.datasets_name}", 'rb') as f:
    dict_ = json.load(f)
  
  if 'bt' == args.run_name: 
    dataset_dict = gen_back_trans(dict_, args.prob_cont, args.prob_ques)
  else:
    raise Exception("Error: Unrecognized Augmentation")

  directory = f"{args.save_dir}/{args.datasets_name}_{args.run_name}_{args.prob_cont}_{args.prob_ques}"
  with open(Path(directory), 'w') as f:
    json.dump(dataset_dict, f)

if __name__ == '__main__':
    main()