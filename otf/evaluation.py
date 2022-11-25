from sklearn import metrics as skm
import numpy as np


def evaluate(predictor, dataset, data_name=''):
    unprot_input, prot_input, target = dataset[:]

    prediction = predictor.predict(unprot_input).numpy()
    prot_input = prot_input.numpy()
    target = target.numpy()

    metrics = {f"auc_{data_name}": skm.roc_auc_score(target, prediction)}

    for prot_name, prot_col_idx in zip(['sex', 'race'], [[0, 1], [2, 3]]):
        prot_input_cols = prot_input[:, prot_col_idx]
        # prot_input_col = np.argmax(prot_input_cols, axis=1)

        metrics[f"pdp_{data_name}_{prot_name}"] = pdp(prediction, prot_input_cols)
        metrics[f"peo_{data_name}_{prot_name}"] = peo(prediction, prot_input_cols, target)
    return metrics


def pdp(preds, prot_input):
    all_corrs = np.corrcoef(preds, prot_input, ddof=0, rowvar=False)

    # Only interested in correlation between preds (first row) and sens vals (second+ cols).
    lin_ind = np.abs(all_corrs[0, 1:])
    max_lin_ind = np.max(lin_ind)
    return max_lin_ind


def peo(preds, prot_input, target):
    max_lin_ind_per_val = []
    for target_val in np.unique(target):
        target_val_idx = target == target_val
        preds_for_val = preds[target_val_idx]
        prot_input_for_val = prot_input[target_val_idx]

        max_lin_ind_for_val = pdp(preds_for_val, prot_input_for_val)
        max_lin_ind_per_val.append(max_lin_ind_for_val)
    max_lin_corr = np.max(max_lin_ind_per_val)
    return max_lin_corr
