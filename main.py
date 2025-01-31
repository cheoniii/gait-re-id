import os
import re
import random
import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
import pandas as pd
import numpy as np
import argparse
from torch.utils.data import DataLoader
from sklearn.model_selection import train_test_split
from collections import defaultdict
from datasets import ContrastiveDataset, CMCDataset # datasets.py 임포트
from sklearn.metrics import roc_auc_score, f1_score, confusion_matrix, accuracy_score, classification_report, roc_curve
from sklearn.metrics.pairwise import cosine_similarity

# RNG 시드 고정
seed = 42
torch.manual_seed(seed)
torch.cuda.manual_seed(seed)
torch.cuda.manual_seed_all(seed)  # if use multi-GPU
np.random.seed(seed)
random.seed(seed)

# CUDA 결정론적 설정
torch.backends.cudnn.deterministic = True
torch.backends.cudnn.benchmark = False
os.environ["PYTHONHASHSEED"] = str(seed)
os.environ["TF_ENABLE_ONEDNN_OPTS"] = "0"

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

def parse_arguments():
    parser = argparse.ArgumentParser(description='Run experiments with different parameters.')
    parser.add_argument('--num_negative_pairs', type=int, default=5, help='Number of negative pairs for ContrastiveDataset.')
    parser.add_argument('--num_epochs', type=int, default=30, help='Number of epochs for training.')
    parser.add_argument('--margin', type=float, default=0.5, help='Margin for Contrastive Loss.')
    parser.add_argument('--num_augmentations', type=int, default=5, help='Number of augmentations')
    parser.add_argument('--transform', type=str, choices=["fft", "stft", "wavelet"],required=True, help="Specify the transformation method: 'fft', 'stft', or 'wavelet'.")
    return parser.parse_args()

def get_labels(data_dir):
    file_names = os.listdir(data_dir)
    pattern = r'case(\d+)_.*_IMU_\d+'
    # pattern = r'Case(\d+)_.*_[A-Za-z]{2,4}_\d{3}_freq\.csv'

    file_paths = []
    labels = []

    for file_name in file_names:
        match = re.match(pattern, file_name)
        if match:
            case_num = int(match.group(1))  # 케이스 번호 추출
            file_path = os.path.join(data_dir, file_name)
            file_paths.append(file_path)
            labels.append(case_num)
        else:
            print(f"파일 이름 패턴과 일치하지 않음: {file_name}")

    return file_paths, labels

def split_case(file_paths, labels, test_cases=None, test_size=0.2, random_seed=42):
    label_to_files = defaultdict(list)
    for fp, label in zip(file_paths, labels):
        label_to_files[label].append(fp)

    unique_labels = list(label_to_files.keys())
    unique_labels.sort()  # Sort by case number

    pattern = re.compile(r'case(\d+)_')
    # Initialize train and test files
    train_files = []
    test_files = []

    if test_cases:
        for path in file_paths:
            match = pattern.search(path)
            if match:
                case_number = int(match.group(1))
                if case_number in test_cases:
                    test_files.append(path)
                else:
                    train_files.append(path)
    else:
        # 원래 random train_test_split
        train_labels, test_labels = train_test_split(
            unique_labels,
            test_size=test_size,
            random_state=random_seed
        )
        for label in train_labels:
            train_files.extend(label_to_files[label])

        for label in test_labels:
            test_files.extend(label_to_files[label])
    return train_files, test_files

class ContrastiveLoss(nn.Module):
    def __init__(self, margin=0.5):
        super(ContrastiveLoss, self).__init__()
        self.margin = margin

    def forward(self, output1, output2, label):
        euclidean_distance = F.pairwise_distance(output1, output2, keepdim=True)
        loss_contrastive = torch.mean(
            (1 - label) * torch.pow(euclidean_distance, 2) +
            label * torch.pow(torch.clamp(self.margin - euclidean_distance, min=0.0), 2)
        )
        return loss_contrastive

class EmbeddingNet(nn.Module):
    def __init__(self, input_channels=1, embedding_dim=128):
        super(EmbeddingNet, self).__init__()
        self.conv = nn.Sequential(
            nn.Conv2d(input_channels, 32, kernel_size=3, padding=1),
            nn.BatchNorm2d(32),
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=2, stride=2),

            nn.Conv2d(32, 64, kernel_size=3, padding=1),
            nn.BatchNorm2d(64),
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=2, stride=2),

            nn.Conv2d(64, 128, kernel_size=3, padding=1),
            nn.BatchNorm2d(128),
            nn.ReLU(),
            nn.AdaptiveAvgPool2d((1, 1))
        )
        self.fc = nn.Sequential(
            nn.Linear(128, embedding_dim),
            nn.ReLU(),
            nn.Linear(embedding_dim, embedding_dim)
        )

    def forward(self, x):
        x = self.conv(x)
        x = x.view(x.size(0), -1)
        x = self.fc(x)
        x = F.normalize(x, p=2, dim=1)
        return x

def train_model(model, train_loader, num_epochs, learning_rate=0.00005, margin=0.5):
    model.to(device)
    criterion = ContrastiveLoss(margin=margin)
    optimizer = optim.Adam(model.parameters(), lr=learning_rate)

    for epoch in range(num_epochs):
        model.train()
        total_loss = 0
        
        for data1, data2, labels in train_loader:
            data1, data2, labels = data1.to(device), data2.to(device), labels.to(device).float()

            optimizer.zero_grad()
            output1 = model(data1)
            output2 = model(data2)
            loss = criterion(output1, output2, labels)
            loss.backward()
            optimizer.step()

            total_loss += loss.item()
        print(f"Epoch [{epoch+1}/{num_epochs}], Loss: {total_loss/len(train_loader):.4f}")

    torch.save(model.state_dict(), 'model_weights.pth')
    print("Model weights saved.")

def load_model(model_path, input_channels=1, embedding_dim=128):
    model = EmbeddingNet(input_channels=input_channels, embedding_dim=embedding_dim)
    model.load_state_dict(torch.load(model_path, map_location=device))
    model.to(device)
    model.eval()
    return model

def extract_embeddings_and_labels(dataloader, model, device):
    """
    ContrastiveDataset 으로 embedding / label 추출
    Args:
        dataloader / model / device
    Returns:
        embedding1 (np.array): 첫 번째 샘플 임베딩
        embedding2 (np.array): 두 번째 샘플 임베딩
        labels (np.array): 각 쌍의 레이블 (1: Positive, 0: Negative)
    """
    embeddings1 = []
    embeddings2 = []
    labels = []
    
    model.eval()
    with torch.no_grad():
        for data1, data2, label in dataloader:
            data1 = data1.to(device)
            data2 = data2.to(device)

            emb1 = model(data1)
            emb2 = model(data2)

            embeddings1.append(emb1.cpu().numpy())
            embeddings2.append(emb2.cpu().numpy())
            labels.extend(label.numpy())

    embeddings1 = np.vstack(embeddings1)
    embeddings2 = np.vstack(embeddings2)
    labels = np.array(labels)

    return embeddings1, embeddings2, labels

def extract_embeddings_and_labels_cmc(dataloader, model, device):
    """
    CMCDataset을 통해 임베딩과 레이블을 추출하는 함수
    Args:
        dataloader (DataLoader) : CMCDataset -> DataLoader
        model (torch.nn.Module)
        device (torch.device)
    Returns:
        embeddings (np.array): test set 임베딩
        labels (np.array): test set 라벨 (case num)
    """
    embeddings = []
    labels = []

    model.eval()
    with torch.no_grad():
        for data, label in dataloader:
            data = data.to(device)
            emb = model(data)
            embeddings.append(emb.cpu().numpy())
            labels.extend(label.numpy())

    embeddings = np.vstack(embeddings)
    labels = np.array(labels)

    return embeddings, labels


def compute_cosine_similarity(embeddings1, embeddings2):
    """
    두 임베딩 간의 Cosine Similarity를 계산하는 함수
    Args:
        embeddings1 (np.array): 첫 번째 샘플의 임베딩
        embeddings2 (np.array): 두 번째 샘플의 임베딩
    Returns:
        similarity (np.array): 각 쌍의 Cosine Similarity
    """
    similarity = np.sum(embeddings1 * embeddings2, axis=1) / (
        np.linalg.norm(embeddings1, axis=1) * np.linalg.norm(embeddings2, axis=1)
    )
    return similarity

def find_optimal_threshold(similarity, labels):
    """
    ROC Curve 기반 optima threshold 찾음
    """
    fpr, tpr, thresholds = roc_curve(labels, similarity)
    J = tpr - fpr
    ix = np.argmax(J)
    optimal_threshold = thresholds[ix]
    print(f"Optimal Threshold: {optimal_threshold:.4f}")
    return optimal_threshold

def evaluate_binary_classification(similarity, labels, threshold=None):
    """
    Binary Classification -> Positive / Negative
    Args:  
        similarity (np.array)
        lables (np.array): 실제 레이블
        threshold (float, optional)
    Returns:
        Metrics (dict)
    """ 
    # AUC score 
    try:
        auc = roc_auc_score(labels, similarity)
    except ValueError:
        auc = None
    
    if threshold is None:
        threshold = find_optimal_threshold(similarity, labels)
    else:
        print(f"Using Provided threshold: {threshold}")
    
    # 예측 레이블 생성
    y_pred = (similarity >= threshold).astype(int)

    # F1 Score 계산
    f1 = f1_score(labels, y_pred)

    # Confusion Matrix 계산
    cm = confusion_matrix(labels, y_pred)

    # Specificity와 Sensitivity 계산
    if cm.shape == (2, 2):
        TN, FP, FN, TP = cm.ravel()
        specificity = TN / (TN + FP) if (TN + FP) > 0 else 0
        sensitivity = TP / (TP + FN) if (TP + FN) > 0 else 0
    else:
        specificity = None
        sensitivity = None

    # Accuracy 계산
    accuracy = accuracy_score(labels, y_pred)

    # Classification Report
    class_report = classification_report(labels, y_pred)

    # 결과 반환
    return {
        'AUC': auc,
        'Threshold': threshold,
        'F1 Score': f1,
        'Specificity': specificity,
        'Sensitivity': sensitivity,
        'Accuracy': accuracy,
        'Classification Report': class_report
    }

def create_query_gallery(embeddings, labels, num_queries_per_class=1, multiple_galleries=False):
    """
    Setting query and gallery
    Args:
        embeddings (np.array) : 모든 샘플의 임베딩
        labels (np.array) : 모든 샘플 labels (=> case num)
        num_queries_per_class : 각 클래스당 쿼리로 사용할 샘플의 수
        multiple_galleries: gallery 2개 사용 여부
    Returns:
        queries (np.array) : 쿼리 임베딩
        query_labels (np.array) : 쿼리 라벨
        galleries (np.array) : 갤러리 임베딩
        gallery_labels (np.array) : 갤러리 라벨
    """
    random.seed(42) 
    label_to_indices = defaultdict(list)
    for idx, label in enumerate(labels):
        label_to_indices[label].append(idx)
    
    queries = []
    query_labels = []
    gallery1 = []
    gallery1_labels = []
    gallery2 = []  # Second gallery if multiple_galleries is True
    gallery2_labels = []

    for label, indices in label_to_indices.items():
        if len(indices) < num_queries_per_class + 1:
            continue    # 설정한 num_queries_per_class 보다 한 클래스 안에 데이터 개수가 부족하면 건너뜀
        query_indices = random.sample(indices, num_queries_per_class)
        remaining_indices = list(set(indices) - set(query_indices))

        for q_idx in query_indices:
            queries.append(embeddings[q_idx])
            query_labels.append(label)
        
        # 갤러리는 한 개만 선택 (single gallery setting)
        # 예: 각 클래스마다 하나의 갤러리 샘플을 선택
        gallery1_idx = random.choice(remaining_indices)
        gallery1.append(embeddings[gallery1_idx])
        gallery1_labels.append(label)

        if multiple_galleries:
            # Choose a second gallery sample, distinct from the first
            remaining_indices.remove(gallery1_idx)
            gallery2_idx = random.choice(remaining_indices)
            gallery2.append(embeddings[gallery2_idx])
            gallery2_labels.append(label)
    
    queries = np.array(queries)
    query_labels = np.array(query_labels)
    gallery1 = np.array(gallery1)
    gallery1_labels = np.array(gallery1_labels)
    
    if multiple_galleries:
        gallery2 = np.array(gallery2)
        gallery2_labels = np.array(gallery2_labels)
        return queries, query_labels, gallery1, gallery1_labels, gallery2, gallery2_labels
    else:
        return queries, query_labels, gallery1, gallery1_labels

def compute_cmc(queries, query_labels, galleries, gallery_labels, top_k=[1, 5]):
    """
    CMC 점수를 계산
    Args:
        queries (np.array) : Query 임베딩
        query_labels (np.array)
        galleries (np.array) : Gallery 임베딩
        gallery_labels (np.array)
        top_k (list): 계산할 k (1 and 5)
    Returns:
        cmc_scores (dict) : Rank별 CMC score
    """
    similarity = cosine_similarity(queries, galleries)

    sorted_indices = np.argsort(-similarity, axis=1) # 내림차순 정렬

    cmc_scores = {k: 0 for k in top_k}
    num_queries = queries.shape[0]

    for i in range(num_queries):
        q_label = query_labels[i]
        sorted_gallery_labels = gallery_labels[sorted_indices[i]]
        rank = np.where(sorted_gallery_labels == q_label)[0]
        if len(rank) == 0:
            continue  # 정답이 갤러리에 없는 경우
        first_rank = rank[0] + 1  # 1-based index
        
        for k in top_k:
            if first_rank <= k:
                cmc_scores[k] += 1
    
    for k in top_k:
        cmc_scores[k] = cmc_scores[k] / num_queries
    
    return cmc_scores

def main():
    args = parse_arguments()
    
    data_dir = '/data/gait_re_id/imu_rel_pelvis'
    test_size = 0.2
    random_seed = 42

    augment = False
    num_negative_pairs_test = 5

    random.seed(random_seed)
    torch.manual_seed(random_seed)
    np.random.seed(random_seed)
    print(data_dir)

    file_paths, labels = get_labels(data_dir)
    test_cases = [222, 19, 280, 303, 182, 4, 63, 48, 3, 21, 8, 112, 201, 198, 130, 139,
                  178, 16, 94, 103, 296, 33, 13, 309, 257, 53, 135, 229, 132, 299, 111, 14,
                  32, 187, 80, 314, 306, 211, 113, 208, 56, 254, 305, 221, 20, 176, 204, 86,
                  29, 15, 249, 243, 193, 55, 235, 101, 301, 144, 7, 106, 185, 308, 11, 281]
    train_files, test_files = split_case(file_paths, labels, test_cases=test_cases, random_seed=random_seed)

    train_labels = [int(re.search(r'case(\d+)_', os.path.basename(fp)).group(1)) for fp in train_files]
    test_labels = [int(re.search(r'case(\d+)_', os.path.basename(fp)).group(1)) for fp in test_files]

    if args.transform == 'fft':
        input_channel = 1
    elif args.transform == 'stft':
        input_channel = 21
    elif args.transform == 'wavelet':
        input_channel = 23
    else:
        raise ValueError(f"Unsupported transform type: {args.transform}")


    train_dataset = ContrastiveDataset(
        file_paths=train_files, 
        labels=train_labels, 
        transform=args.transform, 
        input_channel=input_channel,
        augment=augment,
        num_augmentations=args.num_augmentations, 
        num_negative_pairs=args.num_negative_pairs)
    
    
    test_easy_dataset = ContrastiveDataset(
        file_paths=test_files,
        labels=test_labels,
        transform=args.transform,
        input_channel=input_channel,
        num_negative_pairs=num_negative_pairs_test,
        augment=False,  # 평가 시 증강 비활성화
        test_set_options='easy'
    )

    test_normal_dataset = ContrastiveDataset(
        file_paths=test_files,
        labels=test_labels,
        transform=args.transform,
        input_channel=input_channel,
        num_negative_pairs=num_negative_pairs_test,
        augment=False,  # 평가 시 증강 비활성화
        test_set_options='normal'
    )
    
    test_hard_dataset = ContrastiveDataset(
        file_paths=test_files,
        labels=test_labels,
        transform=args.transform,
        input_channel=input_channel,
        num_negative_pairs=num_negative_pairs_test,
        augment=False,  # 평가 시 증강 비활성화
        test_set_options='hard'
    )

    # overall test set
    test_dataset = ContrastiveDataset(
        file_paths=test_files,
        labels=test_labels,
        transform=args.transform, 
        input_channel=input_channel,
        num_negative_pairs=num_negative_pairs_test,
        augment=False,  # 평가 시 증강 비활성화
        test_set_options=[test_easy_dataset.negative_pair, test_normal_dataset.negative_pair, test_hard_dataset.negative_pair]
    )

    cmc_dataset = CMCDataset(
        file_paths=test_files,
        labels=test_labels,
        transform=args.transform, 
        input_channel=input_channel
    )

    def worker_init_fn(worker_id):
        seed = 42
        np.random.seed(seed + worker_id)
        random.seed(seed + worker_id)

    train_loader = DataLoader(train_dataset, batch_size=64, shuffle=True, num_workers=4, pin_memory=True)
    test_loader = DataLoader(test_dataset, batch_size=32, shuffle=False, num_workers=4, worker_init_fn=worker_init_fn)
    test_easy_loader = DataLoader(test_easy_dataset, batch_size=32, shuffle=False, num_workers=4, worker_init_fn=worker_init_fn)
    test_normal_loader = DataLoader(test_normal_dataset, batch_size=32, shuffle=False, num_workers=4, worker_init_fn=worker_init_fn)
    test_hard_loader = DataLoader(test_hard_dataset, batch_size=32, shuffle=False, num_workers=4, worker_init_fn=worker_init_fn)
    cmc_loader = DataLoader(cmc_dataset, batch_size=32, shuffle=False, num_workers=4, worker_init_fn=worker_init_fn)

    model = EmbeddingNet(input_channels=input_channel, embedding_dim=128)
    train_model(model, train_loader=train_loader, num_epochs=args.num_epochs, learning_rate=1e-3, margin=args.margin)

    model_loaded = load_model('model_weights.pth', input_channels=input_channel, embedding_dim=128)

    # test - (auc, f1 score, ...)

    # 테스트 로더들을 딕셔너리에 저장
    test_loaders = {
        "Overall": test_loader,
        "Easy": test_easy_loader,
        "Normal": test_normal_loader,
        "Hard": test_hard_loader
    }

    # 각 테스트 로더에 대해 평가를 수행하는 루프
    for name, loader in test_loaders.items():
        print(f"\nEvaluating {name} Test Set:")

        # 임베딩과 레이블 추출
        emb1, emb2, labels = extract_embeddings_and_labels(loader, model_loaded, device)
        similarity_scores = compute_cosine_similarity(emb1, emb2)

        # 이진 분류 메트릭 계산
        metrics = evaluate_binary_classification(similarity_scores, labels)

        # 결과 출력
        print("=== Binary Classification Metrics ===")
        print(f"{name} AUC Score: {metrics['AUC']:.4f}" if metrics['AUC'] is not None else "AUC Score: N/A")
        print(f"{name} F1 Score: {metrics['F1 Score']:.4f}")
        print(f"{name} Specificity: {metrics['Specificity']:.4f}" if metrics['Specificity'] is not None else "Specificity: N/A")
        print(f"{name} Sensitivity: {metrics['Sensitivity']:.4f}" if metrics['Sensitivity'] is not None else "Sensitivity: N/A")
        print(f"{name} Accuracy: {metrics['Accuracy']:.4f}")
        print("\nClassification Report:")
        print(metrics['Classification Report'])


    # test - cmc score
    test_embeddings, test_labels = extract_embeddings_and_labels_cmc(cmc_loader, model_loaded, device)

    queries, query_labels, gallery1, gallery1_labels, gallery2, gallery2_labels = create_query_gallery(
        test_embeddings, test_labels, num_queries_per_class=1, multiple_galleries = True
    )

    cmc_scores = compute_cmc(queries, query_labels, gallery1, gallery1_labels, top_k=[1, 5])
    print("\n=== Gallery 1 CMC Scores ===")
    for k, score in cmc_scores.items():
        print(f"Gallery1_Rank-{k}: {score*100:.2f}%")

    cmc_scores = compute_cmc(queries, query_labels, gallery2, gallery2_labels, top_k=[1, 5])
    print("\n=== Gallery 2 CMC Scores ===")
    for k, score in cmc_scores.items():
        print(f"Gallery2_Rank-{k}: {score*100:.2f}%")


if __name__ == '__main__':
    main()