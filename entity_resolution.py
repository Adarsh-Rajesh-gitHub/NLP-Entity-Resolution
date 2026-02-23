import pandas as pd
#regex
import re
from rapidfuzz import fuzz
from sklearn.metrics import precision_recall_fscore_support
from sklearn.feature_extraction.text import TfidfVectorizer
import numpy as np

def normalize_text(text: str) -> str:
    #handle missing values
    if pd.isna(text): return ''
    text = text.lower()
    #removes all characters other then letters and numbers
    text = re.sub(r'[^a-z0-9]', ' ', text)
    #standardize white space by replacing any spacing with just one space
    text = re.sub(r'\s+', ' ', text).strip()
    return text

def build_pairs(pairs_csv: str, amazon_df: pd.DataFrame, google_df: pd.DataFrame) -> pd.DataFrame:
    #load gold-standard pairs for a split and join ids to actual product names
    df = pd.read_csv(pairs_csv)

    #maps the ids and pull the name data from respective sheets
    df['left_raw'] = df['source_id'].map(amazon_df['name'])
    df['right_raw'] = df['target_id'].map(google_df['name'])

    #have a column of 1/0 instead of True/False for easier time with analysis later on
    df['label'] = df['matching'].astype(int)

    #check if anything was not joined properly
    assert df['left_raw'].isna().sum() == 0 and df['right_raw'].isna().sum() == 0

    #normalize text before scoring
    df['left_norm'] = df['left_raw'].apply(normalize_text)
    df['right_norm'] = df['right_raw'].apply(normalize_text)

    return df

def add_fuzzy(df: pd.DataFrame) -> pd.DataFrame:
    #compute rapidfuzz score (0-100)
    df['fuzzy'] = [
        fuzz.token_sort_ratio(a, b)
        for a, b in zip(df['left_norm'], df['right_norm'])
    ]
    return df

def add_tfidf_cos(df: pd.DataFrame) -> pd.DataFrame:
    #compute tf-idf char n-gram cosine similarity (0-1)
    vec = TfidfVectorizer(analyzer='char', ngram_range=(3, 5), min_df=1)

    #fit tf-idf on all strings in this split so vector space is consistent
    all_text = pd.concat([df['left_norm'], df['right_norm']], ignore_index=True)
    X = vec.fit_transform(all_text)

    n = len(df)
    L = X[:n]
    R = X[n:]

    #tf-idf vectors are l2-normalized so dot product = cosine similarity
    df['tfidf_cos'] = (L.multiply(R).sum(axis=1)).A1
    return df

def eval_metrics(df: pd.DataFrame, score_col: str, t: float):
    #predict match if score >= threshold
    preds = (df[score_col] >= t).astype(int)

    #precision/recall/f1
    p, r, f1, _ = precision_recall_fscore_support(
        df['label'], preds, average='binary', zero_division=0
    )

    #accuracy
    acc = float((preds == df['label']).mean())

    #how many predicted matches + how many errors
    pred_pos = int(preds.sum())
    errors = int((preds != df['label']).sum())

    return p, r, f1, acc, pred_pos, errors

def print_threshold_sweep(df: pd.DataFrame, score_col: str, thresholds, label: str):
    print('\n' + label)
    best = (-1, None, None, None, None, None, None)  #(f1, t, p, r, acc, pred_pos, errors)

    for t in thresholds:
        p, r, f1, acc, pred_pos, errors = eval_metrics(df, score_col, float(t))
        print(
            "threshhold: " + str(float(t)) +
            " | accuracy: " + str(round(acc * 100, 2)) +
            " | P: " + str(round(p, 4)) +
            " | R: " + str(round(r, 4)) +
            " | F1: " + str(round(f1, 4)) +
            " | pred_pos: " + str(pred_pos) +
            " | number errors: " + str(errors)
        )

        if f1 > best[0]:
            best = (f1, float(t), p, r, acc, pred_pos, errors)

    print("\nBEST by F1 => threshhold:", best[1],
          "| accuracy:", round(best[4] * 100, 2),
          "| P/R/F1:", round(best[2], 4), round(best[3], 4), round(best[0], 4),
          "| pred_pos:", best[5],
          "| number errors:", best[6])

    return best[1]  #best threshold

#load amazon and google data into data frames, had do to do encoding 'latin-1' because of some utf8 parsing issue
amazon_df = pd.read_csv('1_amazon.csv', encoding='latin-1')
google_df = pd.read_csv('2_google.csv', encoding='latin-1')

#make them searchable by their id instead of pandas inbuilt indexing
amazon_df = amazon_df.set_index('subject_id')
google_df = google_df.set_index('subject_id')

#build train/val/test pair tables
train_df = build_pairs('gs_train.csv', amazon_df, google_df)
val_df = build_pairs('gs_val.csv', amazon_df, google_df)
test_df = build_pairs('gs_test.csv', amazon_df, google_df)

#MODEL A: rapidfuzz only (0-100)
train_df = add_fuzzy(train_df)
val_df = add_fuzzy(val_df)
test_df = add_fuzzy(test_df)

#sweep thresholds like your old code
best_fuzzy_t = print_threshold_sweep(
    val_df,
    score_col='fuzzy',
    thresholds=np.linspace(0.0, 100.0, 101),
    label='FUZZY on VAL (tune threshold here)'
)

#final test metrics at best threshold
p, r, f1, acc, pred_pos, errors = eval_metrics(test_df, 'fuzzy', best_fuzzy_t)
print("\nFUZZY TEST @ best_val_thresh:", best_fuzzy_t,
      "| accuracy:", round(acc * 100, 2),
      "| P/R/F1:", round(p, 4), round(r, 4), round(f1, 4),
      "| pred_pos:", pred_pos,
      "| number errors:", errors)

#MODEL B: tf-idf cosine only (0-1)
train_df = add_tfidf_cos(train_df)
val_df = add_tfidf_cos(val_df)
test_df = add_tfidf_cos(test_df)

best_tfidf_t = print_threshold_sweep(
    val_df,
    score_col='tfidf_cos',
    thresholds=np.linspace(0.0, 1.0, 101),
    label='TFIDF_COS on VAL (tune threshold here)'
)

p, r, f1, acc, pred_pos, errors = eval_metrics(test_df, 'tfidf_cos', best_tfidf_t)
print("\nTFIDF_COS TEST @ best_val_thresh:", best_tfidf_t,
      "| accuracy:", round(acc * 100, 2),
      "| P/R/F1:", round(p, 4), round(r, 4), round(f1, 4),
      "| pred_pos:", pred_pos,
      "| number errors:", errors)