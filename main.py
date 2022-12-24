import pandas as pd
import nltk
from collections import Counter


# nltk.download('averaged_perceptron_tagger')
# nltk.download('punkt')
# nltk.download('universal_tagset')
def get_data_excel_sheets():
    path = 'data/thesis_transcript.xlsx'
    df = pd.read_excel(path, sheet_name=None)
    return df


def get_data_nth_sheet(n):
    path = 'data/thesis_transcript.xlsx'
    df = pd.read_excel(path, sheet_name=n)
    return df


def is_garbage(line):
    if not line:
        return True
    actual_line: str = line.strip()
    return (
            (not actual_line)
            or actual_line.casefold() == 'x'.casefold()
            or actual_line.casefold() == 'nan'.casefold()
            or actual_line.startswith("start_scene")
            or actual_line.startswith("end_scene")
    )


def read_sheet_n(sheetNumber):
    first_sheet = get_data_nth_sheet([sheetNumber])[sheetNumber]
    response_columns = first_sheet['utterances'].astype('str')
    tokenized_utterances = []
    for line in response_columns:
        if not is_garbage(line):
            tokenized_utterances += nltk.tokenize.word_tokenize(line)
    pos_tags = nltk.pos_tag(tokenized_utterances, tagset='universal')
    tag_counts = Counter(tag for word, tag, in pos_tags if tag)
    return tag_counts


def clean_counter(counter: Counter):
    bad_tags = ['X', '.']
    for tag in bad_tags:
        del counter[tag]
    return counter


if __name__ == "__main__":
    autistic_sheets = [0, 1, 2]
    neurotypical_sheets = [3, 4, 5]
    print("AUTISTIC COUNTS:")
    print("___________________")
    autistic_counts = []
    for autistic_sheet in autistic_sheets:
        counter = read_sheet_n(autistic_sheet)
        autistic_counts.append(clean_counter(counter))
    autistic_df = pd.concat([pd.Series(c) for c in autistic_counts], axis=1).T.replace(float('NaN'), 0).astype('int64')
    print(autistic_df)

    nt_counts = []
    print()
    print("NT COUNTS:")
    print("___________________")
    for neurotypical_sheet in neurotypical_sheets:
        counter = read_sheet_n(neurotypical_sheet)
        nt_counts.append(clean_counter(counter))
    nt_df = pd.concat([pd.Series(c) for c in nt_counts], axis=1).T.replace(float('NaN'), 0).astype('int64')
    print(nt_df)
    with pd.ExcelWriter('data/results.xlsx') as writer:
        autistic_df.to_excel(writer, sheet_name='AUT')
        nt_df.to_excel(writer, sheet_name='NT')
