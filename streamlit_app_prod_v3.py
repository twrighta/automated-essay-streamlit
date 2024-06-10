#Prod_v2 - Downloads model from google cloud, turn downloading models into a cached function
#Prod_v3 - adds gdown package to try download the model properly from google cloud
# Import
import joblib as jl
import regex as re
from nltk import flatten, word_tokenize, sent_tokenize
import numpy as np
import pandas as pd
from scipy.stats import iqr, skew, kurtosis
import streamlit as st
import re
import warnings
import time
import textstat
from spellchecker import SpellChecker
import shap
import matplotlib.pyplot as plt
import seaborn as sns
import threading
import requests
import os
import gdown

# Extras?
import lightgbm
import catboost
import nltk


nltk.download('punkt')


warnings.simplefilter('ignore')

#################################################################################################################

# Load in submission model and required transformers from directory
#model_local_path = 'trained_model_v2.joblib'
#model_download_url = 'https://drive.google.com/uc?export=download&id=13QF-QF5b5o6N16z3E4nTZw7NrFEcL8dN'
model_download_url = 'https://drive.google.com/file/d/13QF-QF5b5o6N16z3E4nTZw7NrFEcL8dN/view?usp=drive_link'
tfidf_url = 'tfidf_v2.joblib'
svd_url = 'svd_v2.joblib'
scaler_url = 'robust_scaler_v2.joblib'

st.set_page_config(page_title = 'Essay Grade Prediction',
                   layout = 'wide',
                   page_icon = ':game_die:')

@st.cache_resource
def load_pretrained(tfidf_url, svd_url, scaler_url, model_download_url):
    tfidf = jl.load(tfidf_url)
    svd = jl.load(svd_url)
    scaler = jl.load(scaler_url)


    #Download model with gdown
    downloaded = gdown.cached_download(model_download_url, 'trained_model_v2.joblib')
    print(f'downloaded: {downloaded}')
    #Download joblib model
    model = jl.load(downloaded)

    return tfidf, svd, scaler, model

tfidf, svd, scaler, model = load_pretrained(tfidf_url, svd_url, scaler_url, model_download_url)


# Preprocessing functions - these transform the text input into features in a dataframe to generate prediction from

# Count use of punctuation characters
def count_punctuation(text):
    punctuation = '!"#$%&\'()*+,-./:;<=>?@[\\]^_`{|}~'
    return sum(text.count(char) for char in punctuation)


def count_bad_formatting(text):
    period_regex = r"\.[A-Za-z]+"
    period_regex2 = r"[A-Za-z]+\s\."
    period_regex3 = r"\.\."
    caps_regex = r"\.\s[a-z]+"
    colon_regex = r":[A-Za-z]+"
    colon_regex2 = r":\s[A-Z]+"
    exclamation_regex = r"![A-Za-z]+"
    question_regex = r"\?[A-Za-z]+"
    dash_regex = r"[A-Za-z]+-\s[A-Za-z]"
    double_regex = r"[?!:;,]{2,}"

    regex_list = [period_regex, period_regex2, period_regex3, caps_regex, colon_regex,
                  colon_regex2, exclamation_regex, question_regex, dash_regex, double_regex]
    matches = []
    for regex in regex_list:
        try:
            matches.append(re.findall(regex, text))
        except:
            continue
    matches_flattened = flatten(matches)
    return len(matches_flattened)


def count_unique(text):
    return len(set(text.split(' ')))

# Count sentences longer than length words, indicative of complex sentences and better quality.
def count_long_sentences(text, length):
    sentences = re.split(r'[.!?:]+', text)  # Split text into sentences
    long_sentences = [sentence.strip() for sentence in sentences if
                      len(sentence.split()) > length]  # Filter sentences longer than length
    return len(long_sentences)


# Count sentences shorter than length words
def count_short_sentences(text, length):
    sentences = re.split(r'[.!?:]+', text)
    short_sentences = [sentence.strip() for sentence in sentences if
                       len(sentence.split()) < length]  # Filter sentences shorter than length
    return len(short_sentences)

def sentence_character_length_kurtosis(text):
    sentences = re.split(r'[.!?:]+', text)  # Split text into sentences by . ! ? occurences
    sentence_lens = np.array([len(sentence.strip()) for sentence in sentences])
    np.insert(sentence_lens, -1, [2, 2, 2])
    return kurtosis(sentence_lens)

def sentence_character_length_iqr(text):
    sentences = re.split(r'[.!?:]+', text)  # Split text into sentences by . ! ? occurences
    sentence_lens = np.array([len(sentence.strip()) for sentence in sentences])
    np.insert(sentence_lens, -1, [2, 2, 2])
    return iqr(sentence_lens)

def sentence_character_length_skew(text):
    sentences = re.split(r'[.!?:]+', text)  # Split text into sentences by . ! ? occurences
    sentence_lens = np.array([len(sentence.strip()) for sentence in sentences])
    np.insert(sentence_lens, -1, [2, 2, 2])
    return skew(sentence_lens)


# Count sentences with extremely long lengths - these are difficult to read. e.g., over 35 words
def count_extreme_sentences(text):
    sentences = re.split(r'[.!?:]+', text)  # Split text into sentences
    long_sentences = [sentence.strip() for sentence in sentences if
                      len(sentence.split()) > 35]
    return len(long_sentences)

# Count long sentences with no commas or : or ; or -
def count_long_sentences_no_punct(text):
    sentences = re.split(r'[.!?:]+', text)
    puncts = [",", ";", "-"]
    long_sentences_no_punct = []
    long_sentences = [sentence.strip() for sentence in sentences if len(sentence.split()) > 30]
    for sentence in long_sentences:
        if all(char not in puncts for char in sentence):
            long_sentences_no_punct.append(sentence)
    return len(long_sentences_no_punct)


def average_word_length(text):
    words = text.split(' ')
    word_lengths = [len(word) for word in words]
    return np.median(word_lengths)


# Count ands, indicative of e poor quality sentences when used too frequently.
def count_ands(text):
    words = text.split(' ')
    and_count = [word.strip() for word in words if word == 'and']
    return len(and_count)


# Count overused words of little value. Indicative of lower quality writing.
def count_obsolete_words(text):
    obsolete_words = ['basically', 'actually', 'absolutely', 'certainly', 'totally', 'really', 'very', 'that', 'just', 'think',
                      'got', 'so', 'thing', 'things', 'something', 'stuff', 'like', 'always', 'never', 'big',
                      'important', 'maybe', 'generally', 'whatever', 'interesting', 'amazing', 'often',
                      'seriously' 'someone', 'huge', 'best', 'worst', 'nice',
                      'true', 'seems', 'believe', 'find', 'found', 'seemed', 'get', 'usually', 'many', 'bad',
                      'mostly']
    words = text.split(' ')
    obsolete_matches = [word.strip() for word in words if word in obsolete_words]
    obsolete_count = len(obsolete_matches)
    return obsolete_count


# Simple to complex sentences ratio
def simple_complex_sentences(text, len_complex):
    sentences = re.split(r'[.!?]+', text)
    complex_punctuation = [':', ';', '-']
    complex_clauses = ['after', 'although', 'because', 'before', 'if', 'though', 'unless', 'when', 'whilst',
                       'since', 'even though', 'rather', 'contrarily', 'however', 'contrastingly', 'contrast',
                       'opposingly', 'conversely']

    complex_sentences = [sentence.strip() for sentence in sentences if (len(sentence.split(' ')) > len_complex) or
                         (sentence.split(' ') in (complex_punctuation))
                         or (sentence.split(' ') in (complex_clauses))]

    simple_sentences_len = len(sentences) - len(complex_sentences)
    return (simple_sentences_len + 1) / (len(complex_sentences) + 1)


# Count transitional phrases - indicative of longer flowing texts / arguments
def count_transitional_phrases(text):
    words = text.split(' ')
    transitional_phrases = ['additionally', 'furthermore', 'moreover', 'likewise', 'consequently', 'accordingly',
                            'therefore', 'thus', 'nonetheless', 'conversely', 'meanwhile', 'subsquently', 'similarly',
                            'nonetheless', 'hence', 'albeit', 'whereupon', 'henceforth', 'inasmuch']
    trans_matches = [word.strip() for word in words if word in transitional_phrases]
    trans_count = len(trans_matches)
    return trans_count


# Common typos in english language - indicative of poorly written/reviewed texts
def count_common_typos(text):
    words = text.split(' ')
    common_typos = ['becuase', 'concensus', 'akward', 'embarassed', 'recieve', 'successfull',
                    'diffrent', 'recieved', 'intresting', 'acommodate', 'acknowledgement', 'aquire',
                    'apparant', 'calender', 'coleague', 'experiance', 'fulfil', 'indispensible',
                    'layed', 'liasion', 'licence', 'maintainance', 'neccessary', 'occassion', 'occured',
                    'pastime', 'privelege', 'priviledge', 'publically', 'pubicly', 'recomend', 'reccommend',
                    'refered', 'relevent', 'seperate', 'succesful', 'successfull', 'underate', 'untill',
                    'withold', 'beleive', 'enterpreneur', 'thourough', 'arguement', 'accomodate',
                    'belive', 'bizzare', 'collegue', 'cheif', 'definately', 'dissapoint', 'goverment',
                    'happend', 'mischeivous', 'theif', 'wierd', 'consscioius', 'occassion', 'guage', 'fro',
                    'mthe', 'seperate', 'wich', 'recieve', 'occured', 'definately', 'pharoah', 'acceptble',
                    'particuly', 'paralell', 'suport', 'beautifull', 'liason', 'tommorrow', 'unforseen',
                    'reminice', 'fortunatly', 'unfortunatly', 'accross', 'agression', 'remeber', 'seige',
                    'tatoo', 'jist', 'apparantly', 'teh', 'A', 'cuase']
    typo_matches = [word.strip() for word in words if word in common_typos]
    typo_count = len(typo_matches)
    return typo_count


def short_long_word_ratio(text):
    words = text.split(' ')
    punct = [".", ",", "!", "?"]
    short_words = len([word for word in words if len(word) < 4])
    long_words = len([word for word in words if len(word) > 7])

    try:
        return short_words / long_words
    except ZeroDivisionError:
        return 1


# log Average paragraph length - words
def log_average_paragraph_length(text):
    paragraphs = re.split(r'\n\n|\n|\n\n\n', text)
    paragraph_lens = []

    for paragraph in paragraphs:
        words = paragraph.split(' ')
        paragraph_lens.append(len(words))

    return np.log(np.mean(np.array(paragraph_lens)))


# Average sentence count per paragraph:
def average_sentence_count_paragraph(text):
    paragraphs = re.split(r'\n\n|\n|\n\n\n', text)

    for paragraph in paragraphs:
        para_sentence_list = []
        sentence_count = 0
        for char in paragraph:
            if char in [".", ":", "!", "?"]:
                sentence_count += 1
        para_sentence_list.append(sentence_count)

    return np.mean(np.array(para_sentence_list))


# Passive voice use - this is often an indication of weaker writing.
def passive_voice_usage(text):
    passive_voice_list = ['am', 'is', 'are', 'were', 'be', 'being', 'been',
                          'was', 'being', 'been', 'become', 'became', 'becoming',
                          'appears', 'appeared', 'appear', 'seem', 'seemed', 'feel',
                          'seems', 'felt', 'look', 'feels', 'sounded', 'tasted',
                          'smelled', 'heard', 'hear', 'touch', 'touched', 'known',
                          'thought', 'supposed', 'believed', 'those', 'will', 'have',
                          'had', 'has', 'shall', 'can', 'might', 'may', 'should',
                          'must', 'must\'ve']
    words = text.split(' ')
    passive_matches = len([word.strip() for word in words if word in passive_voice_list]) + 0.001
    return passive_matches / len(words)

# Text stat features (these require pip install so not in the original kaggle notebook must load model
def flesch_reading_ease(text):
    return textstat.flesch_reading_ease(text)

def dalechall_readability(text):
    return textstat.dale_chall_readability_score(text)

def reading_time(text):
    return textstat.reading_time(text)

def percentage_difficult_words(text):
    return textstat.difficult_words(text) / textstat.lexicon_count(text)

def coleman_liau_index(text):
    return textstat.coleman_liau_index(text)

# Spelling checking
spellchecker_obj = SpellChecker(language = 'en')
def incorrect_spelling(text, spellchecker_obj):
    words = word_tokenize(text)
    incorrect_words = spellchecker_obj.unknown(words)
    try:
        return len(incorrect_words) / len(words)
    except ZeroDivisionError:
        return 1


def incorrect_spelling_sentences_std(text, spellchecker_obj):
    sentences = sent_tokenize(text)
    sentence_incorrects = []
    for sentence in sentences:
        incorrect_words = 0
        for word in word_tokenize(sentence):
            if spellchecker_obj.unknown([word]):
                incorrect_words += 1
        sentence_incorrects.append(incorrect_words)
    return np.std(sentence_incorrects)


# Preprocess input text function! Output a dataframe of features.
def calculate_text_features(text, spellchecker):
    df = pd.DataFrame()
    df['input_text'] = [text]

    # Simple calculations
    df['total_character_length'] = df['input_text'].str.count('.')
    df['number_words'] = df['input_text'].str.count('\s+') + 1
    df['number_sentences'] = df['input_text'].str.count('\.') + 1
    df['number_paragraphs'] = df['input_text'].str.count('\n') + 1
    df['number_punctuations'] = df['input_text'].apply(count_punctuation) + 1
    df['unique_elements'] = df['input_text'].apply(count_unique)

    # Moderately intensive calculations
    df['avg_sentence_length'] = df['number_words'] / df['number_sentences']
    df['sentence_char_length_kurtosis'] = df['input_text'].apply(sentence_character_length_kurtosis)
    df['sentence_char_length_iqr'] = df['input_text'].apply(sentence_character_length_iqr)
    df['sentence_char_length_skew'] = df['input_text'].apply(sentence_character_length_skew)

    # Ensure no nans are returned from above:
    df.fillna(0, inplace = True)
    df.replace(-np.inf, 0, inplace = True)
    df.replace(np.inf, 0, inplace = True)


    df['avg_word_length'] = df['input_text'].apply(average_word_length)
    df['avg_punctuation_per_sentence'] = df['number_punctuations'] / df['number_sentences']
    df['number_long_sentences'] = df['input_text'].apply(
        lambda x: count_long_sentences(x, 19))
    df['number_short_sentences'] = df['input_text'].apply(
        lambda x: count_short_sentences(x, 12))
    df['number_extreme_sentences'] = df['input_text'].apply(count_extreme_sentences)
    df['long_no_punct_sentences'] = df['input_text'].apply(count_long_sentences_no_punct)
    df['and_count'] = df['input_text'].apply(count_ands)
    df['obsolete_word_count'] = df['input_text'].apply(count_obsolete_words)
    df['formatting_mistakes'] = df['input_text'].apply(count_bad_formatting)
    df['simple_complex_sentences'] = df['input_text'].apply(lambda x: simple_complex_sentences(x, 8))
    df['transitional_count'] = df['input_text'].apply(count_transitional_phrases)
    df['common_typo_count'] = df['input_text'].apply(count_common_typos)
    df['short_long_word_ratio'] = df['input_text'].apply(short_long_word_ratio)
    df['log_average_paragraph_length'] = df['input_text'].apply(log_average_paragraph_length)
    df['average_sentence_count_paragraph'] = df['input_text'].apply(average_sentence_count_paragraph)
    df['passive_voice_use'] = df['input_text'].apply(passive_voice_usage)

    # Text stats using textstats - not in Kaggle
    df['flesch'] = df['input_text'].apply(flesch_reading_ease)
    df['dalechall'] = df['input_text'].apply(dalechall_readability)
    df['read_time'] = df['input_text'].apply(reading_time)
    df['percentage_difficult_words'] = df['input_text'].apply(percentage_difficult_words)
    df['coleman_liau'] = df['input_text'].apply(coleman_liau_index)

    # Spell check - not in Kaggle
    df['ratio_incorrect_spelling'] = df['input_text'].apply(lambda x: incorrect_spelling(x, spellchecker))
    df['sentence_incorrect_spelling_std'] = df['input_text'].apply(
        lambda x: incorrect_spelling_sentences_std(x, spellchecker))


    # Creating proportioned columns
    df['proportion_non_punctuation'] = ((df['total_character_length'] - df['number_punctuations']) / df[
        'total_character_length']) * 100
    df['proportion_ands'] = ((df['and_count'] + 1) / df['number_words']) * 100
    df['proportion_obsolete'] = ((df['obsolete_word_count'] + 1) / df['number_words']) * 100
    df['proportion_formatting_error'] = ((df['formatting_mistakes'] + 1) / df['number_punctuations'])
    df['transitional_ratio'] = ((df['transitional_count'] + 1) / (df['number_words'] + 1))
    df['common_typo_ratio'] = ((df['common_typo_count'] + 1) / (df['number_words'] + 1))

    # Dropping excess columns no longer needed after creating the proportioned ones:
    df.drop(columns = ['and_count', 'obsolete_word_count', 'formatting_mistakes',
                       'transitional_count', 'common_typo_count'], inplace = True)

    # Ensure no nans are returned from above:
    df.fillna(0, inplace = True)
    df.replace(-np.inf, 0, inplace = True)
    df.replace(np.inf, 0, inplace = True)

    # return dataframe for transform and predict function to transform
    return df



# Apply transformations from training, and output prediction
def transform_and_predict(df, model, tfidf, svd, scaler):

    # Transform input_text
    tfidf_input = tfidf.transform(df['input_text'])

    # Decompose tfidf vectorized text and return results as dataframe to concatenate.
    decomp_tfidf_input = pd.DataFrame(svd.transform(tfidf_input))

    # Concatenate the decomposed tfidf output to the main dataframe as new columns
    full_df = pd.concat([df.reset_index(drop = True), decomp_tfidf_input], axis = 1)

    # Drop the input text column
    full_df.drop(columns = ['input_text'], inplace = True)

    # Make sure all column headers are type strings - the tfidf outputs are integers originally.
    full_df.columns = full_df.columns.astype(str)
    feature_names = full_df.columns

    # Apply scaling transformation to the full_df
    full_df_scaled = scaler.transform(full_df)

    text_features = full_df_scaled

    # Generate predictions
    raw_pred = model.predict(text_features)

    # Clip, round, and output final predicted grade
    clipped_pred = np.clip(raw_pred,
                           a_min = 1,
                           a_max = 6)

    final_pred = np.round(clipped_pred)

    return final_pred, full_df_scaled, feature_names

# ShaP plots using voting classifiers not single models seem to work poorly.
def shap_feature_importance(X, model, feature_names, predicted_class_idx):
    shap_explainer = shap.KernelExplainer(model.predict, X)
    shap_values = shap_explainer(X)
    # print(f'\nshap_values.values: {shap_values.values}')

    shap_summary = shap.plots.force(shap_explainer.expected_value,
                                    shap_values[predicted_class_idx, :],
                                    features = feature_names,
                                    matplotlib = True,
                                    show = False)
    plt.show()

    return shap_summary

# Function to loop through the tree-based estimators within voting classifier, calculate shap values,
# put into df, plot on expandable barchart with st.expander() to explain what the plot shows.
def treebased_shap(X, model, feature_names):
    model_names = list(model.named_estimators_.items())
    estimator_shap_df = pd.DataFrame()
    estimator_shap_df['features'] = feature_names

    for i, estimator in enumerate(model.estimators_):
        if hasattr(estimator, 'max_depth'):
            estimator_explainer = shap.TreeExplainer(estimator,
                                                X,
                                                feature_names = feature_names)
            estimator_shap = pd.Series(estimator_explainer(X).data.flatten())
            estimator_name = str(model_names[i]).split(',')[0].replace("\'","").strip("(")
            estimator_shap_df[estimator_name] = estimator_shap
        else:
            pass
    # Calculate average feature importance across all models as a new column
    estimator_shap_df['avg_shap'] = estimator_shap_df.iloc[:, 1:].mean(axis = 1)
    estimator_shap_df['abs_avg_shap'] = abs(estimator_shap_df['avg_shap'])

    # Order by largest absolute importance (positive or negative)
    ordered_shap_df = estimator_shap_df.sort_values(by = 'abs_avg_shap', ascending = False).copy()

    # Select top 10 features by absolute importance
    top_ordered_shap_df = ordered_shap_df.head(10)

    # Create a matplotlib figure to return plot
    fig, ax = plt.subplots()
    sns.barplot(data = top_ordered_shap_df,
                x = 'features',
                y = 'avg_shap')
    plt.xticks(rotation = 45)
    plt.rc('xtick', labelsize = 8)
    plt.ylabel('Average SHAP Value')
    plt.xlabel('Feature')
    plt.title('Most important features')
    plt.tight_layout()

    return fig


#################################################################################################
# Streamlit Application

def update_shap_progress(shap_progress_bar):
    for i in range(101):
        time.sleep(0.18)
        shap_progress_bar.progress(i, 'Calculating SHAP values. Please wait')

def run_shap_calculation():
    global estimator_shap_plot
    estimator_shap_plot = treebased_shap(X_df, model, feature_names)


st.set_option('deprecation.showPyplotGlobalUse', False) # Disable pyplot warning
#st.set_page_config(page_title = 'Essay Grade Prediction',
#                   layout = 'wide',
#                   page_icon = ':game_die:')
st.title('Predict an essay\'s grade using machine learning! :game_die:')
st.subheader('Essays are graded 1-6, ranging from poor to outstanding')
st.divider()
essay = st.text_area('Please paste an essay below:')
st.divider()

if essay:


    if st.button('Generate Predictions'):

        # Calculate text features
        output_df = calculate_text_features(essay, spellchecker_obj)

        # Transform and predict.
        predictions, X_df, feature_names = transform_and_predict(output_df, model, tfidf, svd, scaler)

        progress_bar = st.progress(0, 'Processing. Please wait.')
        for i in range(101):
            time.sleep(0.185)
            progress_bar.progress(i, 'Processing. Please wait.')
        time.sleep(5)
        progress_bar.empty()

        # Remove done box:
        success = st.success('Done')
        success.empty()
        # Display predictions
        st.metric('Predicted Grade:', predictions[0])
        st.divider()

        shap_progress_bar = st.progress(0, 'Calculating SHAP values')

        # Create and start SHAP calculation thread simulataneously
        shap_calc_thread = threading.Thread(target = run_shap_calculation)
        shap_calc_thread.start()

        update_shap_progress(shap_progress_bar)

        # Join progress bar and calculation threads together once complete
        shap_calc_thread.join()

        # Plot shap in expander
        with st.expander('See plot and explanation'):
            st.write('''Larger SHAP values indicate larger influence on outputs. 
                Positive SHAP values contribute a positive increase in probability of 
                predicting a certain class. Negative SHAP values indicate a feature is contributing
                to a decrease in the predicted probability of a certain class.
                ''')
            st.divider()
            st.pyplot(estimator_shap_plot)

        time.sleep(2)
        shap_progress_bar.empty()