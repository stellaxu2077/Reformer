import pandas as pd
import numpy as np
import h5py
import transformers as T
import torch
from tqdm import tqdm
from utils import load_model_bc
from train_reformer_bc import SequenceDataset, Bert4BinaryClassification  # Import trained model class

np.random.seed(42)


tokenizer = T.BertTokenizer.from_pretrained("./model/") 
model_path = "./model/model_best.bin"  # Use best model checkpoint
model, device = load_model_bc(tokenizer, model_path)

test_h5file = "./data/example_posneg.h5"  # Path to test data
dataset = SequenceDataset(test_h5file, tokenizer, mode="test")

predictions = []
batch_size = 32  # Adjust as needed

for idx in tqdm(range(0, len(dataset), batch_size), desc="Processing Batches"):
    batch_inputs = [dataset[i] for i in range(idx, min(idx + batch_size, len(dataset)))]
    batch_inputs = torch.cat(batch_inputs, dim=0).to(device)  # Concatenate inputs
    
    with torch.no_grad():
        batch_outputs = model(batch_inputs).squeeze().detach().cpu().numpy()
    
    predictions.extend(batch_outputs)

output_df = pd.DataFrame({"Sequence_ID": np.arange(len(predictions)), "Prediction": predictions})
output_df.to_csv("test_predictions.csv", index=False)

print("Predictions saved to test_predictions.csv")
print(output_df[:5])

