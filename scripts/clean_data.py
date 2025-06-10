"""
The purpose of this document is to preprocess the transcript dataset.

(1) tokenize_text converts an individual transcript into its tokenized form.
(2) convert_date converts the dataset dates into pandas DateTime objects.
(3) remove_duplicates does just that, removing duplicate listings in the dataset. 

Finally, update_dataset creates a copy of the motley-fool-data, first calling function (3) to minimize 
the work needed, then calling (2) and (1), updating each entry.

We use this script to create a tokenized version of the Motley Fool transcripts for use by bow.ipynb.

Decided to do rest of project in Jupyter notebooks for quicker implementation and subjectively feeling "closer" to the data. 
"""

from nltk.corpus import stopwords
from nltk.stem import WordNetLemmatizer

import re
import pandas as pd

def tokenize_text(text, upper):
    """Upper dictates whether tokens should be upper or lower case"""
    # Assume Q&A section separated by "Questions & Answers:"
    text = text.split("Questions and Answers:")

    if len(text) == 2:
    # Only consider Q&A data
        text = text[1]

    else: 
        text = text[0].split("Questions & Answers:")
        
        if len(text) == 2:
            text = text[1]

        else: text = text[0] # Simplifying assumption: Take entire text if no separate Q&A section. 

    # Separate punctuation from words
    text = re.sub(r'[^\w\s]+', ' ', text)
    
    # Split by whitespace
    if upper:
        words = text.upper().split()
    else: 
        words = text.lower().split()
        
    # Remove stop words
    stop_words = set(stopwords.words("english"))
    words = [word for word in words if word not in stop_words]

    # Lemmatize
    # lemmatized_words = lemmatization(words) 
    """
    LM-dict isn't lemmatized, so skip this step for
    bow data. (as in https://github.com/cancan-huang/Sentiment-Analysis-for-Financial-Articles)
    """

    return words

def convert_dates(document):
    """
    Input must be initial dataframe containing "date", "ticker" columns.
    """

    dates_fixed_df = document.copy(deep=True)

    date_val = []

    for val in document["date"]:
        if type(val) == str:
            date_val.append(val)
        elif type(val) == list:
            if len(val) == 1:
                date_val.append(val[0].split("Call")[1])
            else:
                date_val.append(val[1])

    date = pd.DataFrame(date_val)

    dates_fixed_df["Origin_Date"] = date[0]

    dates_fixed_df["Origin_Date"] = dates_fixed_df["Origin_Date"].astype(str)

    dates_fixed_df = dates_fixed_df[dates_fixed_df["Origin_Date"] != ""]

    dates_fixed_df["Origin_Date"] = dates_fixed_df["Origin_Date"].apply(lambda val:val.replace('p.m.', 'PM').replace('a.m.', 'AM').replace('ET', '').strip())
    
    # Fix strange outlier
    dates_fixed_df.loc[dates_fixed_df["Origin_Date"] == 'TranscriptMarch 13, 2018, 8:30 AM', 'Origin_Date'] = 'March 13, 2018, 8:30 AM'

    dates_fixed_df["Origin_Date"] = dates_fixed_df["Origin_Date"].apply(pd.to_datetime)

    # Also, for convenience, clean ticker here
    dates_fixed_df['ticker'] = dates_fixed_df['ticker'].str.replace('[^a-zA-Z0-9]', '', regex=True)

    return dates_fixed_df

# def lemmatization(texts):
#     texts_out = []
#     lemmatizer = WordNetLemmatizer()

#     for sent in texts:
#         for word in sent:
#             texts_out.append(lemmatizer.lemmatize(word))
#     return texts_out

def remove_duplicates(document):
    """Remove all rows where ticker and date are the same"""
    return document.drop_duplicates(subset = ["Origin_Date", "ticker"], inplace = False)

def update_dataset(tokenize=False):
    initial_dataset = pd.read_pickle("..\data\motley-fool-data.pkl")
    
    dates_fixed_df = convert_dates(initial_dataset)

    no_duplicates = remove_duplicates(dates_fixed_df)
    no_duplicates = no_duplicates.reset_index(drop=True)

    cleaned__text = []
    
    if tokenize:
        for i in range(len(no_duplicates)):
        # for i in range(1):
            # Tokenize the text and store it in a variable
            transcript = no_duplicates["transcript"][i]
            cleaned__text.append(tokenize_text(transcript, upper=False))
        no_duplicates["cleaned_transcript"] = (cleaned__text)

    preprocessed_df = no_duplicates.drop(["date","exchange"], axis = 1)

    return(preprocessed_df)

"""
Run below to update csvs with 1) bow tokenized data or 2) date fixed and duplicates removed.
Next, we load the returns dataset, match it with the columns of the cleaned transcripts 
and export this as a "finished" dataset with all the rows we will use for the classification. 
"""
# if __name__ == "__main__":
#     preprocessed_df = update_dataset(tokenize=True)
#     print(preprocessed_df.head())

#     preprocessed_df = preprocessed_df.drop("transcript", axis = 1)

#     preprocessed_df.to_csv("..\data\BoW-tokenized-transcripts.csv")

"""
Remove comments to create either cleaned-dataset or BoW-tokenized-transcripts
"""

if __name__ == "__main__":
    preprocessed_df = update_dataset()

    preprocessed_df.to_csv("..\data\cleaned-dataset.csv")
    print("Done")