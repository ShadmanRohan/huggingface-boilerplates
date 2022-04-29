# huggingface-boilerplates
Starter code for HuggingFace NLP-pipelines

Basic Training Code
```
args = TrainingArguments(num_train_epochs, per_device_train_batch_size, per_device_eval_batch_size, warmup_steps, ...)
trainer = Trainer(model, train_data, eval_data, compute_metrics, args=args)
trainer.train()
```

For Predicting
```
trainer.predict(test_dataset)
```

Pulling model from HUB
```
from transformers import pipeline

model_name = "example-distilbert-base-uncased"
tokenizer = DistilBertTokenizer.from_pretrained(model_name, do_lower_case=True)
model_id = "UserName/RepoName"
classifier = pipeline("text-classification", tokenizer= tokenizer, model=model_id)

```

Things to remember
* compute_metrics(pred) -> returns a dictionary of metrics like accuracy, f1, precision, recall
* dataset -> pytorch dataset(not dataloader). The data needs to be tokenized & structured for input to transformers.
