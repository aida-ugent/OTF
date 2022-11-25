import random
import numpy as np
import pandas as pd
import torch
from torch.utils.data import TensorDataset, DataLoader
from aif360.datasets import AdultDataset
from sklearn.preprocessing import StandardScaler

from otf.predictor import Predictor
from otf.evaluation import evaluate
from otf.linear_fairness_notion import ProbabilisticDemographicParity, ProbabilisticEqualizedOdds
from otf_cost import OTFCost


def main():
    seed = 0
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)

    # Load raw training data and split into 80% train and 20% test data
    full_df = AdultDataset().convert_to_dataframe()[0]
    rand_idx = np.random.permutation(np.arange(len(full_df)))
    split_idx = int(0.8 * len(full_df))
    train_df = full_df.iloc[rand_idx[:split_idx]]
    test_df = full_df.iloc[rand_idx[split_idx:]]

    # Preprocess raw data into PyTorch datasets
    datasets = []
    scaler = StandardScaler()
    for i, df in enumerate([train_df, test_df]):
        prot_feat = pd.get_dummies(df[['sex', 'race']].astype(str)).values
        target = df['income-per-year'].values
        unprot_feat = df.drop(columns=['sex', 'race', 'income-per-year']).values

        if i == 0:
            unprot_feat = scaler.fit_transform(unprot_feat)
        else:
            unprot_feat = scaler.transform(unprot_feat)

        dataset = TensorDataset(
            torch.from_numpy(unprot_feat).float(),
            torch.from_numpy(prot_feat).float(),
            torch.from_numpy(target).long()
        )
        datasets.append(dataset)
    train_data, test_data = datasets

    # Initialise and fit model on the data
    model = Predictor(nb_features=train_data.tensors[0].shape[1],
                      nb_epochs=100)
    fairness_notion = ProbabilisticEqualizedOdds()
    otf_cost = OTFCost(fairness_notion,
                       nb_epochs=100,
                       margin_tol=1e-3,
                       constraint_tol=1e-3,
                       reg_strength=1e-3)
    train_dataloader = DataLoader(train_data,
                                  batch_size=1000,
                                  shuffle=True,
                                  drop_last=False,
                                  collate_fn=otf_cost.collate_fn)
    model.fit(train_dataloader, otf_cost)

    # Evaluate
    print(evaluate(model, train_data, data_name="train"))
    print(evaluate(model, test_data, data_name="test"))


if __name__ == '__main__':
    main()
