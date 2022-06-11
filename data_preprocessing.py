import encodings
from tqdm import tqdm
import numpy as np # linear algebra
import pandas as pd
import os
from sklearn.model_selection import train_test_split

num_classes = 20

def fetch_data():
    '''
    Read and clean data
    '''
    df = pd.read_csv("booksummaries/booksummaries.txt", delimiter="\t", encoding='utf-8')
    df.columns = ['Wikipedia Article ID', 'Firebase ID', 'Book title', 'Author', 'Publication date', 'Book genres', 'Plot Summary']
    df.dropna(subset=['Book genres'],inplace = True)


    all_genres = []
    book_genres = []
    for item in tqdm(df['Book genres'], desc="Fetch data"):
        n = len(item)
        genres = item[1:n-1].split(',')
        genre_list = [genre.split(':')[1].strip() for genre in genres]
        genre_list = [genre[1:len(genre)-1] for genre in genre_list]
        book_genres.append(genre_list)

        for i in genre_list:
            if i not in all_genres:
                all_genres.append(i)

    df['Genres'] = book_genres

    return df, all_genres

def filter_genres(df, all_genres_list):
    genre_counts = dict()
    for item in df['Genres']:
        for genre in item:
            if genre not in genre_counts.keys():
                genre_counts[genre] = 1
            else:
                genre_counts[genre] += 1
    
    genres = []
    for key, val in genre_counts.items():
        genres.append((key, val))

    genres.sort(key = lambda x: x[1])
    filters = [i[0] for i in genres[-num_classes:]]
    return filters

def clean_labels(df, filtered_genres):
    new_genres = []
    for item in tqdm(df['Genres'], desc="Clean labels"):
        new_item = []
        for genre in item:
            if genre in filtered_genres:
                new_item.append(genre)
        new_genres.append(new_item)
    df['Filtered Genres'] = new_genres

    count = 0
    lens = []
    for item in df['Filtered Genres']:
        lens.append(len(item))
        if len(item) == 0:
            count += 1
    print("Count: " + str(count))
    print("Max: " + str(max(lens)))
    print("Min: " + str(min(lens)))

    return df

def encode(df, all_genres_list):
    all_genres_list.sort()
    n = len(all_genres_list)

    encoded_genres = []
    
    for item in tqdm(df['Genres'], desc="Encoding"):
        item.sort()
        encoding = ""
        for category in all_genres_list:
            if category in item:
                encoding += "1"
            else:
                encoding += "0"
        encoded_genres.append(encoding)
    df["Encoded Genres"] = np.array(encoded_genres, dtype = str)
    df = df.astype(str)
    return df, encoded_genres


def create_splits(X_train, y_train, X_val, y_val, X_test, y_test):
    split_path = "DataSplit"
    if not os.path.exists(split_path):
        os.makedirs(split_path)
    else:
        print("Directory already exists!")

    train_df_X = pd.DataFrame(X_train)
    train_df_Y = pd.DataFrame(y_train)
    train_df = pd.concat([train_df_X, train_df_Y], axis = 1)
    train_df.to_csv(os.path.join(split_path, "train.tsv"), sep="\t", index=False)

    val_df_X = pd.DataFrame(X_val)
    val_df_Y = pd.DataFrame(y_val)
    val_df = pd.concat([val_df_X, val_df_Y], axis = 1)
    val_df.to_csv(os.path.join(split_path, "dev.tsv"), sep="\t", index=False)

    test_df_X = pd.DataFrame(X_test)
    test_df_Y = pd.DataFrame(y_test)
    test_df = pd.concat([test_df_X, test_df_Y], axis = 1)
    test_df.to_csv(os.path.join(split_path, "test.tsv"), sep="\t", index=False)

def create_data():
    '''
    Split into train and test sets, store in files
    '''
    
    df, all_genre_list = fetch_data()

    filtered_genres = filter_genres(df, all_genre_list)
    print(filtered_genres)
    s = input()
    df = clean_labels(df, filtered_genres)
    
    df, encoded_genres = encode(df, filtered_genres)

    print(encoded_genres[:5], len(encoded_genres[0]))
    # s = input()

    X_train, X_test, y_train, y_test = train_test_split(df['Encoded Genres'], df['Plot Summary'], test_size=0.20)
    X_train, X_val, y_train, y_val = train_test_split(X_train, y_train, test_size=0.125)


    create_splits(X_train, y_train, X_val, y_val, X_test, y_test)
    

if __name__ == "__main__":
    create_data()
