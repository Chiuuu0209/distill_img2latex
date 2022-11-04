import math
import time
import sys
import numpy as np
from pathlib import Path
import warnings

import torch
from torch.utils.data import Dataset, DataLoader
from albumentations.pytorch import ToTensorV2
from scipy.special import softmax
from PIL import Image

from model.img2latex.data.utils import get_all_formulas, get_split, BaseDataset
from model.img2latex.lit_models import LitResNetTransformer
warnings.filterwarnings("ignore")


class Newmodel(LitResNetTransformer):
    def __init__(self, *args,**kwargs):
        LitResNetTransformer.__init__(self, *args,**kwargs)

    def BS(self, x,top_k):
        B = x.shape[0]
        S = self.model.max_output_len
    
        encoded_x = self.model.encode(x)  # (Sx, B, E)

        output_indices = torch.full((B, S), self.model.pad_index).type_as(x).long()
        output_indices[:, 0] = self.model.sos_index        
        
        output_sequences = [(output_indices, 0)]
        for Sy in range(1,S):
            new_sequences = []
            for old_seq, old_score in output_sequences:
                y = old_seq[:, :Sy]
                if self.model.eos_index in old_seq[:, :Sy]:
                    new_sequences.append((old_seq, old_score))
                    continue
                logits = self.model.decode(y, encoded_x)
                nplogits = logits.clone().detach().numpy().flatten()
                nplogits = nplogits[-545:]
                probs = softmax(nplogits)
                
                idx = np.argpartition(probs, -top_k)
                max_idx = idx[-top_k:]             
                
                for i in range(top_k):
                    temp_seq = old_seq.clone()
                    temp_seq[:, Sy] = torch.tensor(max_idx[i])
                    new_seq = temp_seq
                    new_score = old_score + math.log(probs[max_idx[i]])
                    new_sequences.append((new_seq, new_score))

            #sort all new sequences in the de-creasing order of their score
            output_sequences = sorted(new_sequences, key = lambda val: val[1], reverse = True)
            
            #select top-k based on score 
            # *Note- best sequence is with the highest score
            output_sequences = output_sequences[:top_k]
            has_end = 0
            for old_seq, old_score in output_sequences:
                if self.model.eos_index in old_seq:
                    has_end +=1
            if has_end == top_k:
                break
        return output_sequences

def BS_data(model_path):
    model = Newmodel.load_from_checkpoint(model_path)

    model.model.eval()

    # calculate num of parameters
    print('Number of parameters: ', sum(p.numel() for p in model.model.parameters() if p.requires_grad))

    data_path = Path(__file__).resolve().parents[0] / "data" 
    formulas_path = data_path / "im2latex_formulas.norm.new.lst"
    filter_path = data_path / "im2latex_train_filter.lst"
    all_formulas = get_all_formulas(formulas_path)
    train_image_names, train_formulas = get_split(all_formulas, filter_path)
    processed_images_path = data_path / "formula_images_processed"
    
    
    
    dis_list = []
    c = 0
    for idx in range(len(train_formulas)):
        train_dataset = BaseDataset(processed_images_path,[train_image_names[idx]], [train_formulas[idx]], ToTensorV2())
        train_dataloader = DataLoader(train_dataset, shuffle=False, batch_size=1, num_workers=4, pin_memory=False)
        for (X, y) in train_dataloader:
            tops = model.BS(X.float(),1)
            for seq,score in tops:
                vocab_list = seq.tolist()
                for vocab in vocab_list:
                    decoded = model.tokenizer.decode(vocab)
                    seq_dis= " ".join(decoded)
            dis_list.append(seq_dis)
    with open(data_path / "new_BS_data.lst", "w") as outfile:
        outfile.write("\n".join(dis_list))

if __name__ == "__main__":
    model_path = sys.argv[1]
    print("Start")
    start = time.time()
    BS_data(model_path)
    end = time.time()
    print(f"Produce BS data Time: {end - start}")