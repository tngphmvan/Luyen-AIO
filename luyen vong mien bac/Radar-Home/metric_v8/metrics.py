import torch
import json
import os
import sys
from submission_model import MyModel
sys.path.append('/bohr/metric-k06r/v8')
from dataloader import load_data, generate_file_paths, CustomDataset
from torch.utils.data import DataLoader

def cal_accuracy(model, test_loader, bonus):
    model.eval()
    total_score = 0
    total_theo = 0
    
    with torch.no_grad():
        for images, labels, _ in test_loader:
            images = images.cuda() if torch.cuda.is_available() else images
            labels = labels.cuda() if torch.cuda.is_available() else labels
            
            outputs = model(images)
            outputs = torch.argmax(outputs, dim=1)
            
            equal_mask = outputs == labels  # correctly predicted masks
            neg_one_mask = labels == 0      # Mask of background categories
            
            # Calculate the score
            score_neg_one = (equal_mask & neg_one_mask).sum() * 1  # Background category score
            score_other = (equal_mask & ~neg_one_mask).sum() * bonus  # Non-background category score
            score_theo = neg_one_mask.sum() * 1 + (~neg_one_mask).sum() * bonus  # Full marks in theory
            
            total_score += score_neg_one + score_other
            total_theo += score_theo
    
    score = total_score.item() / total_theo.item()
    return score

if __name__ == '__main__':
    if os.environ.get('DATA_PATH'):
        DATA_PATH = os.environ.get("DATA_PATH")
    else:
        #验证集 用于选手调试
        DATA_PATH = ""
    base_path_public = DATA_PATH + '/validation_set'
    base_path_private = DATA_PATH + '/testing_set'
    bonus = 5000  # Score weights for non-background categories

    model = MyModel()
    model.load_state_dict(torch.load("submission_dic.pth"))
    
    if torch.cuda.is_available():
        model = model.cuda()
    model.eval()
    
    # load testing_set_public
    val_paths = generate_file_paths(base_path_public)
    val_dataset = CustomDataset(file_paths=val_paths)
    val_loader = DataLoader(
        val_dataset,
        batch_size=1,  
        shuffle=False,
        num_workers=2
    )
    
    # load testing_set_private
    test_paths = generate_file_paths(base_path_private)
    test_dataset = CustomDataset(file_paths=test_paths)
    test_loader = DataLoader(
        test_dataset,
        batch_size=1,  
        shuffle=False,
        num_workers=2
    )
    # Calculate the score
    score_public = cal_accuracy(model, val_loader, bonus)
    if score_public > 1:
        score_public = 0
    print(f"Public set score: {score_public:.4f}")
    
    score_private = cal_accuracy(model, test_loader, bonus)
    if score_private > 1:
        score_private = 0
    print(f"Private set score: {score_private:.4f}")
    
    # Save the results
    score = {
        "public_a": score_public,
        "private_b": score_private,
    }
    ret_json = {
        "status": True,
        "score": score,
        "msg": "Success!",
    }
    with open('score.json', 'w') as f:
        f.write(json.dumps(ret_json))