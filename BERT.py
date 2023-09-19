from transformers import AutoTokenizer

#Using BERT tokenizer
tokenizer = AutoTokenizer.from_pretrained("distilbert-base-uncased")

#Encode the datasets
train_encodings = tokenizer(X_train.tolist(), truncation=True, padding=True)
test_encodings = tokenizer(X_test.tolist(), truncation=True, padding=True)
val_encodings = tokenizer(X_val.tolist(), truncation=True, padding=True)


#Turns encodings into usable dataset
import torch
class TweetsDataset(torch.utils.data.Dataset):
    def __init__(self, encodings, labels = None):
        self.encodings = encodings
        self.labels = labels

    def __getitem__(self, idx):
        item = {key: torch.tensor(val[idx]) for key, val in self.encodings.items()}
        if self.labels:
            item['labels'] = torch.tensor(self.labels[idx])
        return item

    def __len__(self):
        return len(self.encodings["input_ids"])
    

train_dataset = TweetsDataset(train_encodings, list(y_train))
test_dataset = TweetsDataset(test_encodings, list(y_test))
val_dataset = TweetsDataset(val_encodings, list(y_val))


from tensorflow.keras.optimizers import Adam
from transformers import AutoModelForSequenceClassification
from transformers import TrainingArguments, Trainer
import evaluate


labels = np.array(train["veracity"])

#Create mdel
model = AutoModelForSequenceClassification.from_pretrained("distilbert-base-uncased", num_labels = 2)
training_args = TrainingArguments(output_dir="test_trainer", evaluation_strategy = 'epoch', push_to_hub = True)

#Find evaluation Metrics
metric_acc = evaluate.load("accuracy")
metric_recall = evaluate.load("recall")
metric_precision = evaluate.load("precision")
metric_f1 = evaluate.load("f1")
def compute_metrics(eval_pred):
    logits, labels = eval_pred
    predictions = np.argmax(logits, axis=-1)
    return metric_acc.compute(predictions=predictions, references=labels), metric_recall.compute(predictions=predictions, references=labels), metric_precision.compute(predictions=predictions, references=labels), metric_f1.compute(predictions=predictions, references=labels)


def compute_accuracy(eval_pred):
    logits, labels = eval_pred
    predictions = np.argmax(logits, axis=-1)
    return (predictions == labels).mean().item()

def model_init():
  return AutoModelForSequenceClassification.from_pretrained("distilbert-base-uncased", num_labels=2)


#Push model to HuggingFace API
#token: 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
from huggingface_hub import notebook_login
notebook_login()

from transformers import Trainer

#Run model
trainer = Trainer(model = model,
                  #model_init = model_init,
                  args = training_args,
                  compute_metrics=compute_metrics,
                  train_dataset = train_dataset,
                  eval_dataset = test_dataset,
                  tokenizer = tokenizer
)

trainer.push_to_hub()
tokenizer.push_to_hub("test_trainer")

predictions = trainer.predict(test_dataset)
logits, _, _ = predictions

def label_with_threshold(raw_pred, threshold):
  pred = scipy.special.softmax(raw_pred, axis = 1)
  labeled = pred[:,1] > threshold
  return labeled

predicted_labels = label_with_threshold(logits, 0.5)
rough_accuracy = predicted_labels == y_test

#Plot Results
skplt.metrics.plot_confusion_matrix(y_test, predicted_labels, title = " ", labels = [0,1])
plt.show()