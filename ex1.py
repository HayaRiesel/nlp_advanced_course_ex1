import argparse

import numpy as np
import torch
from datasets import load_dataset
from transformers import AutoTokenizer, AutoModelForSequenceClassification, Trainer, TrainingArguments, set_seed, \
    DataCollatorWithPadding, pipeline
# import wandb
import evaluate

def fine_tune_model(model_name, num_seeds, num_train_samples, num_val_samples):

    tokenizer = AutoTokenizer.from_pretrained(model_name)
    model = AutoModelForSequenceClassification.from_pretrained(model_name, num_labels=2)

    dataset = load_dataset("sst2")
    if num_train_samples > 0:
        dataset['train'] = dataset['train'].select(range(num_train_samples))
    if num_val_samples > 0:
        dataset['validation'] = dataset['validation'].select(range(num_val_samples))

    def tokenize_fn(example):
        return tokenizer(example["sentence"], padding=False, truncation=True)

    train_dataset = dataset['train'].map(tokenize_fn, batched=True)
    val_dataset = dataset['validation'].map(tokenize_fn, batched=True)

    data_collator = DataCollatorWithPadding(tokenizer)

    training_args = TrainingArguments(
        output_dir="results",
        logging_dir="logs",
        # report_to="wandb",
        save_strategy="epoch",
        save_total_limit=1,
        logging_steps=1000,
        eval_steps=1000
    )

    # Fine-tuning loop
    accuracies = []
    train_runtime = 0
    eval_runtime = 0
    for seed in range(num_seeds):
        set_seed(seed)

        # run = wandb.init(project="ex1", config={"model": model_name, "seed": seed})
        metric = evaluate.load("accuracy")
        def compute_metrics(eval_pred):
            logits, labels = eval_pred
            predictions = np.argmax(logits, axis=-1)
            return metric.compute(predictions=predictions, references=labels)

        trainer = Trainer(
            model=model,
            args=training_args,
            train_dataset=train_dataset,
            eval_dataset=val_dataset,
            data_collator=data_collator,
            compute_metrics=compute_metrics
        )

        #train the model
        train_result = trainer.train()
        train_runtime += train_result.metrics['train_runtime']

        #evaluation the model
        model.eval()
        eval_results = trainer.evaluate()
        eval_runtime += eval_results['eval_runtime']

        accuracy = eval_results["eval_accuracy"]
        accuracies.append(accuracy)

        #save the model
        if '/' in model_name:
            model_name = model_name.replace('/', '-')
        path_save = f"{model_name}_seed_{seed}"
        trainer.save_model(path_save)

        # run.finish()

    # compute mean and standard
    mean_accuracy = torch.tensor(accuracies).mean().item()
    std_accuracy = torch.tensor(accuracies).std().item()

    return mean_accuracy, std_accuracy, train_runtime, eval_runtime

def predict(model_name, best_seed, num_test_samples):
    dataset = load_dataset("sst2")
    if num_test_samples > 0:
        dataset['test'] = dataset['validation'].select(range(num_test_samples))

    tokenizer = AutoTokenizer.from_pretrained(model_name)
    # Tokenize dataset
    def tokenize_fn(example):
        return tokenizer(example["sentence"], padding=False, truncation=True)

    test_dataset = dataset['test'].map(tokenize_fn, batched=True)


    if '/' in model_name:
        model_name = model_name.replace('/', '-')
    pipe = pipeline("text-classification", model=f"{model_name}_seed_{best_seed}", tokenizer=tokenizer)
    predictions = pipe(test_dataset["sentence"])

    return test_dataset['sentence'], predictions

def write_res_file(models, mean_accuracy, std_accuracy, train_runtime, eval_runtime):
    with open('res.txt', 'w') as file:
        for i, model_name in enumerate(models):
            mean = mean_accuracy[i]
            std = std_accuracy[i]
            file.write(f'{model_name},{mean} +- {std}\n')
        file.write('----\n')
        file.write(f'train time,{train_runtime}\n')
        file.write(f'predict time,{eval_runtime}\n')

def write_predictions_file(input_sentences, predictions):
    with open("predictions.txt", "w") as file:
        for i, prediction in enumerate(predictions):
            sentence = input_sentences[i]
            predicted_label = prediction['label'][-1]
            file.write(f'{sentence}###{predicted_label}\n')

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("num_seeds", type=int, help="Number of seeds to be used for each model.")
    parser.add_argument("num_train_samples", type=int, help="Number of samples to be used during training.")
    parser.add_argument("num_val_samples", type=int, help="Number of samples to be used during validation.")
    parser.add_argument("num_test_samples", type=int,
                        help="Number of samples for which the model will predict sentiment.")
    args = parser.parse_args()

    models = ["bert-base-uncased", "roberta-base", "google/electra-base-generator"]
    mean_accuracies = []
    std_accuracies = []
    train_runtime = 0
    eval_runtime = 0
    for model_name in models:
        mean_accuracy, std_accuracy, model_train_runtime, model_eval_runtime = fine_tune_model(model_name, args.num_seeds, args.num_train_samples,
                                                                                               args.num_val_samples)
        mean_accuracies.append(mean_accuracy)
        std_accuracies.append(std_accuracy)
        train_runtime += model_train_runtime
        eval_runtime += model_eval_runtime


    # select the best model
    best_model_index = mean_accuracies.index(max(mean_accuracies))
    best_model_name = models[best_model_index]
    best_seed = best_model_index

    #run the prediction
    input_sentence, predictions = predict(best_model_name, best_seed, args.num_test_samples)
    write_predictions_file(input_sentence, predictions)

    write_res_file(models, mean_accuracies, std_accuracies, train_runtime, eval_runtime)
