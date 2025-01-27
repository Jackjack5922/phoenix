import argparse
import mlflow
import os
import torch
from datetime import datetime
from transformers import BertTokenizer

from get_db import load_data_from_mysql
from split_data import split_dataset
from model_utils import (
    BERTClassifier, 
    create_data_loaders, 
    setup_gpu, 
    train_model
)

# Constants
LABELS = ['CENSURE', 'IMMORAL_NONE', 'HATE', 'DISCRIMINATION', 'SEXUAL', 'ABUSE', 'VIOLENCE', 'CRIME']
MLFLOW_URI = f"file:{os.path.abspath('./mlruns')}"
EXPERIMENT_NAME = "BERT-Multilingual-Classification"

def parse_arguments():
    parser = argparse.ArgumentParser(description='BERT Multilingual Classification Pipeline')
    # Database arguments
    parser.add_argument('--host', required=True, help='MySQL host address')
    parser.add_argument('--database', required=True, help='Database name')
    parser.add_argument('--user', required=True, help='Database user')
    parser.add_argument('--password', required=True, help='Database password')
    parser.add_argument('--table', required=True, help='Table name containing all data')
    
    # Training arguments
    parser.add_argument('--batch_size', type=int, default=32, help='Batch size for training')
    parser.add_argument('--epochs', type=int, default=5, help='Number of epochs')
    parser.add_argument('--learning_rate', type=float, default=2e-5, help='Learning rate')
    
    # Data split arguments
    parser.add_argument('--train_size', type=dict, default={
        'IMMORAL_NONE': 4000, 'CENSURE': 4000, 'SEXUAL': 4000,
        'DISCRIMINATION': 4000, 'HATE': 4000, 'VIOLENCE': 3000,
        'ABUSE': 3000, 'CRIME': 700
    }, help='Number of samples per label for training')
    
    return parser.parse_args()

def setup_mlflow():
    mlflow.set_tracking_uri(MLFLOW_URI)
    mlflow.set_experiment(EXPERIMENT_NAME)
    return mlflow.start_run(
        run_name=f"BERT_Multilingual_classification_{datetime.now().strftime('%Y%m%d_%H%M%S')}"
    )

def main():
    args = parse_arguments()
    
    # 1. Load data from database
    print("Loading data from database...")
    db_config = {
        'host': args.host,
        'database': args.database,
        'user': args.user,
        'password': args.password
    }
    raw_data = load_data_from_mysql(db_config, args.table)
    if raw_data is None:
        print("Failed to load data from database. Exiting...")
        return
        
    # 2. Split data into train/val/test sets
    print("Splitting dataset...")
    train_df, val_df, test_df = split_dataset(raw_data)
    
    # 3. Initialize tokenizer and create data loaders
    print("Creating data loaders...")
    tokenizer = BertTokenizer.from_pretrained('bert-base-multilingual-cased')
    train_loader, val_loader, test_loader = create_data_loaders(
        train_df, val_df, test_df, 
        tokenizer, 
        args.batch_size
    )

    # 4. Initialize model
    print("Initializing model...")
    model = BERTClassifier(len(LABELS))
    device = torch.device('cuda' if setup_gpu() else 'cpu')
    model.to(device)

    # 5. Training pipeline
    print("Starting training pipeline...")
    with setup_mlflow():
        mlflow.log_params({
            "batch_size": args.batch_size,
            "epochs": args.epochs,
            "learning_rate": args.learning_rate,
            "model_type": "bert-base-multilingual-cased",
            "num_labels": len(LABELS)
        })

        try:
            model.load_state_dict(torch.load("model.pkl"))
            print("Loaded pre-trained model.")
        except FileNotFoundError:
            print("No pre-trained model found. Starting training...")
            train_model(
                model=model,
                train_loader=train_loader,
                val_loader=val_loader,
                epochs=args.epochs,
                learning_rate=args.learning_rate,
                device=device
            )

if __name__ == "__main__":
    main()