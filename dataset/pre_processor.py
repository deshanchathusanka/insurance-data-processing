import pandas as pd
from sklearn.preprocessing import LabelEncoder

if __name__ == '__main__':
    # original dataset
    df = pd.read_csv('insurance.csv')

    # data encoding
    label_enc = LabelEncoder()
    df['sex_encoded'] = label_enc.fit_transform(df['sex'])
    df['smoker_encoded'] = label_enc.fit_transform(df['smoker'])
    df['region_encoded'] = label_enc.fit_transform(df['region'])
    df.to_csv('encoded_insurance.csv', index=False)

    # preprocessed dataset
    pre_processed_dataset = df[['age','sex_encoded','bmi','children','smoker_encoded','region_encoded','charges']]
    pre_processed_dataset.to_csv('processed_insurance.csv', index=False)

