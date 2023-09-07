import pandas as pd
from transformers import BertTokenizer, BertForSequenceClassification
from transformers import Trainer, TrainingArguments
from sklearn.model_selection import train_test_split

# Load data
data = pd.read_csv('data/raw/risk_data.csv')

# Tokenize the transaction details
tokenizer = BertTokenizer.from_pretrained('bert-base-uncased')
tokenized_data = tokenizer(list(data['transaction_detail']), truncation=True, padding=True, max_length=128)

# Convert labels to integer format: "not risky" -> 0, "risky" -> 1
labels = data['risk'].apply(lambda x: 0 if x == "not risky" else 1).values

# Split data into training and testing
train_inputs, test_inputs, train_labels, test_labels = train_test_split(tokenized_data.input_ids, labels, test_size=0.2)

# Initialize the model
model = BertForSequenceClassification.from_pretrained('bert-base-uncased', num_labels=2)  # Binary Classification

# Define training arguments and create Trainer
training_args = TrainingArguments(
    per_device_train_batch_size=8,
    num_train_epochs=3,
    logging_dir='./logs',
)

trainer = Trainer(
    model=model,
    args=training_args,
    train_dataset=list(zip(train_inputs, train_labels)),
    eval_dataset=list(zip(test_inputs, test_labels))
)

# Train the model
trainer.train()

# Evaluate the model
results = trainer.evaluate()

print(results)

