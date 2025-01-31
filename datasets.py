# datasets.py
import random
import torch
import pandas as pd
import numpy as np
from torch.utils.data import Dataset
from collections import defaultdict
from preprocessing import reshape_data, augment_data, FFT, STFT, generate_heatmaps
import torch.nn.functional as F


random_seed = 42
random.seed(random_seed)
torch.manual_seed(random_seed)
np.random.seed(random_seed)

class BaseDataset(Dataset):
    def __init__(self, file_paths, transform, input_channel, augment=False, target_length=200):
        """
        csv에서 데이터를 읽어와서 channel수에 따라 처리하는 코드로, 
        모든 Dataset에 동일하게 작용하여 BaseDataset이라고 명명
        Args:
            file_paths: 파일 경로 리스트
            input_channel: 입력 채널 수
        """
        self.file_paths = file_paths
        self.transform = transform
        self.input_channel = input_channel
        self.file_to_idx = {file: idx for idx, file in enumerate(self.file_paths)}  # 인덱스 매핑 최적화
        self.augment = augment
        self.target_length = target_length  # target_length 저장
        self.labels_list = None

    def __len__(self):
        return len(self.file_paths)

    def load_raw_data(self, file_path):

         #  imu 원본 데이터 로드 (csv)
        df = pd.read_csv(file_path, header=0)
        df.replace(0, np.nan, inplace=True)
        df.interpolate(method='linear', inplace=True, limit_direction='both')
        # df = df.applymap(lambda x: np.abs(complex(x)) if isinstance(x, str) else x)        
        df = df.apply(lambda col: col.map(lambda x: np.abs(complex(x)) if isinstance(x, str) else x))

        data = df.values
        data = reshape_data(data)   # shape (sequence length, # joints, 3)

        return data
    

    
    def process_data(self, data):
        """
        데이터를 전처리하고 크기를 통일합니다.
        """
        data = data.reshape(data.shape[0], -1)  # shape: (sequence length, features)

        if self.transform == 'fft':
            data = FFT(data)
            if self.input_channel == 1:
                data = data[np.newaxis, :, :]  # Add channel dimension
        elif self.transform == 'stft':
            data = STFT(data)
        elif self.transform == 'wavelet':
            data = generate_heatmaps(data)
        else:
            data = data.transpose(1, 0)  # Default transpose

        data = self.normalize_data(data)

        # 데이터 크기를 목표 길이로 통일
        data = self.pad_data(data, self.target_length)

        return data.astype(np.float32)
    

    def load_data(self, file_path, augment=False):
        """
        Data load하고, augmentation 및 FFT 적용하는 Method
        Args:
            file_path
            augment
        Returns:
            torch tensor
        """
        data = self.load_raw_data(file_path)
        if augment:
            data = augment_data(data) # aumgmentation -> ContrastiveDataset 에서만 사용
        data = self.process_data(data)
        return torch.tensor(data)
    
    def __getitem__(self, idx):
        file_path = self.file_paths[idx]
        data = self.load_data(file_path, augment=False) # BaseDataset에서는 augmentation 적용하지 않음

        if self.labels_list is not None:
            label = self.labels_list[idx]
            return data, label  # 데이터와 라벨을 반환
        else:
            return data
        # if self.transform:
        #     data = self.transfrom(data)
        
        return data

    def normalize_data(self, data):
        return (data - np.min(data)) / (np.max(data) - np.min(data)) if np.max(data) != np.min(data) else data

class ContrastiveDataset(BaseDataset):
    """
    Contrastive Learning을 위한 데이터셋 클래스
    Positive / Negative pairs 생성
    """
    def __init__(self, file_paths, labels, transform, input_channel, num_negative_pairs=2, augment=False, num_augmentations=2, test_set_options=None, seed=42, target_length=200):
        """
        Args:
            file_paths: 파일 경로 리스트 (data root directory 포함)
            labels: 각 파일의 라벨 리스트
            transform: FFT / STFT / Wavelet Transform
            input_channel: 입력 채널 수
            num_negative_pairs: 각 positive pair에 대해 생성할 negative pair의 수
            augment: 데이터 증강 여부
            num_augmentations: 각 샘플당 생성할 증강 쌍의 수
            test_set_options: 테스트 세트 옵션
        """
        super().__init__(file_paths, transform, input_channel, augment=augment, target_length=target_length)
        self.labels_list = labels
        self.num_negative_pairs = num_negative_pairs
        self.num_augmentations = num_augmentations
        self.augment = augment
        self.test_set_options = test_set_options
        
        # 난수 시드 고정
        random.seed(seed)
        np.random.seed(seed)
        torch.manual_seed(seed)

        self.label_to_indices = self.create_label_to_indices()

        # 카테고리 쌍 초기화
        self.category_pairs = {}
        # 테스트 세트 옵션이 설정된 경우, 카테고리별 음성 쌍을 불러옴
        if self.test_set_options:
            self.category_pairs = self.load_categorized_pairs("./data/filtered_similarity_df.csv")
        self.pairs, self.labels_pairs = self.create_pairs()

    def create_label_to_indices(self):
        label_to_indices = defaultdict(list)
        for idx, label in enumerate(self.labels_list):
            label_to_indices[label].append(idx)
        return label_to_indices

    def create_pairs(self):
        positive_pairs = []
        negative_pairs = []
        unique_labels = list(self.label_to_indices.keys())
        
        for label in unique_labels:
            indices = self.label_to_indices[label]

            # augment 있을 경우, 음성 쌍 비율 조정
            adjusted_num_negative_pairs = self.num_negative_pairs * (self.num_augmentations if self.augment else 1)
            
            if self.augment:
                if len(indices) < 1:
                    continue  # 최소 1개의 샘플이 필요
            else:
                if len(indices) < 2:
                    continue

            if self.augment:
                # 증강된 샘플 생성 및 양의 쌍 생성
                for idx in indices:
                    for _ in range(self.num_augmentations):
                        i, j = random.sample(indices, 2)
                        positive_pairs.append((i, j, 1))
            else:
                # 원본 샘플 간 양의 쌍 생성
                for i in range(len(indices)):
                    for j in range(i + 1, len(indices)):
                        positive_pairs.append((indices[i], indices[j], 1))
            
            # 음의 쌍 생성 (augment 여부와 무관하게 동일하게 처리)
            for idx in indices:
                for _ in range(adjusted_num_negative_pairs):
                    # overall test set (test_set_option이 list로 들어오는 경우)
                    if isinstance(self.test_set_options, list):
                        continue
                    # 단일 test_set_options
                    elif self.test_set_options:  
                        category = self.test_set_options
                        pair = random.choice(self.category_pairs[category])
                        idx1, idx2 = pair[0], pair[1]
                        negative_idx_1 = random.choice(self.label_to_indices[idx1])
                        negative_idx_2 = random.choice(self.label_to_indices[idx2])
                        negative_pairs.append((negative_idx_1, negative_idx_2, 0))
                    # test_set_options=None
                    else:
                        negative_label = random.choice(list(set(unique_labels) - set([label])))
                        negative_idx = random.choice(self.label_to_indices[negative_label])
                        negative_pairs.append((idx, negative_idx, 0))
        
        
        # 음수 쌍을 클래스 속성에 저장
        self.negative_pair = negative_pairs

        # overall test set
        if isinstance(self.test_set_options, list):
            for test_sets in self.test_set_options:
                negative_pairs += test_sets
        
                
        pairs = positive_pairs + negative_pairs
        if self.augment is False:
            if isinstance(self.test_set_options, list):
                print("overall")
            else:
                print(self.test_set_options)
            print("전체 pair 수:",len(pairs))
            print("positive pair 수:",len(positive_pairs))
            print("negative pair 수:",len(negative_pairs),"\n")

        random.shuffle(pairs)
        labels = [pair[2] for pair in pairs]
        return pairs, labels

    def load_categorized_pairs(self, file_path):
        """
        CSV 파일을 읽어 카테고리별로 필터링하여 딕셔너리로 반환
        """
        df = pd.read_csv(file_path)
        easy_pairs = df[df['Category'] == 'Easy'][['Person1', 'Person2']].astype(int).values.tolist()
        normal_pairs = df[df['Category'] == 'Normal'][['Person1', 'Person2']].astype(int).values.tolist()
        hard_pairs = df[df['Category'] == 'Hard'][['Person1', 'Person2']].astype(int).values.tolist()
        return {
            'easy': easy_pairs,
            'normal': normal_pairs,
            'hard': hard_pairs
        }
    
    def load_data_with_augmentation(self, idx):
        """
        데이터에 증강을 적용하여 로드하는 메서드
        Args:
            idx: 파일 인덱스
        Returns:
            torch 텐서, shape: (채널, 특징 수, 시퀀스 길이 또는 FFT 주파수)
        """
        file_path = self.file_paths[idx]
        data = self.load_data(file_path, augment=True)  # 증강 적용
        return data

    '''
    def pad_data(self, data, target_length):
        """
        데이터를 패딩하여 동일한 크기로 만듭니다.
        Args:
            data: torch.Tensor, shape = (채널, 특징 수, 시퀀스 길이 또는 FFT 주파수)
            target_length: 패딩 이후의 목표 길이
        Returns:
            torch.Tensor, 패딩된 데이터
        """
        current_length = data.shape[-1]
        if current_length < target_length:
            # 데이터의 마지막 차원(target_length)에 대해 패딩
            padding = (0, target_length - current_length)  # (왼쪽, 오른쪽)
            data = F.pad(data, padding, mode="constant", value=0)
        return data
    '''
    
    def pad_data(self, data, target_length):
        """
        데이터를 패딩하여 동일한 길이로 만듭니다.
        """
        current_length = data.shape[-1]
        if current_length < target_length:
            padding = (0, target_length - current_length)  # (왼쪽, 오른쪽)
            data = F.pad(data, padding, mode="constant", value=0)
        elif current_length > target_length:
            data = data[:, :target_length]  # 초과된 부분은 잘라냄
        return data


    def __getitem__(self, idx):
        idx1, idx2, label = self.pairs[idx]
        data1 = self.load_data_with_augmentation(idx1)
        data2 = self.load_data_with_augmentation(idx2)
        
        target_length = max(data1.shape[-1], data2.shape[-1])  # 최대 길이를 기준으로 패딩
        
        data1 = self.pad_data(data1, target_length)
        data2 = self.pad_data(data2, target_length)
        
        return data1, data2, torch.tensor(label, dtype=torch.float32)
    
class CMCDataset(BaseDataset):
    def __init__(self, file_paths, labels, transform, input_channel):
        super().__init__(file_paths, transform, input_channel, augment=False)
        self.labels_list = labels

    def __getitem__(self, idx):
        data, label = super().__getitem__(idx)
        return data,label