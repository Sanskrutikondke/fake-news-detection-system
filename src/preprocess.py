import pandas as pd
import re
import string

def clean_text(text):
    text = text.lower()
    text = re.sub('\[.*?\]', '', text)
    text = re.sub("\\W"," ",text) 
    text = re.sub('https?://\S+|www\.\S+', '', text)
    text = re.sub('<.*?>+', '', text)
    text = re.sub('[%s]' % re.escape(string.punctuation), '', text)
    text = re.sub('\n', '', text)
    text = re.sub('\w*\d\w*', '', text)    
    return text

def prepare_data(true_path, fake_path):
    # Load data
    df_true = pd.read_csv(true_path)
    df_fake = pd.read_csv(fake_path)

    # Add labels
    df_true['label'] = 0  # Real
    df_fake['label'] = 1  # Fake

    # Merge and shuffle
    df = pd.concat([df_true, df_fake], axis=0).sample(frac=1).reset_index(drop=True)
    
    # Clean text (combining title and text for better context)
    df['content'] = (df['title'] + " " + df['text']).apply(clean_text)
    
    return df[['content', 'label']]

if __name__ == "__main__":
    combined_df = prepare_data('data/raw/True.csv', 'data/raw/Fake.csv')
    combined_df.to_csv('data/processed/combined_data.csv', index=False)
    print("Data processed and saved to data/processed/combined_data.csv")