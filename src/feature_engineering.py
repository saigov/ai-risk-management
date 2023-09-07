import pandas as pd

def add_features(input_path, output_path):
    data = pd.read_csv(input_path)
    data['token_count'] = data['transaction_detail'].apply(lambda x: len(x.split()))
    data['avg_token_length'] = data['transaction_detail'].apply(lambda x: sum(len(word) for word in x.split()) / len(x.split()))
    data.to_csv(output_path, index=False)

