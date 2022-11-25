import torch
from sklearn import metrics as skm
import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np
import pandas as pd


# Maybe make separate evaluator for every dataset?
from sklearn.metrics import roc_auc_score


class Evaluator:
    def evaluate(self, predictor, dataset, log_plots=True, title_suffix=''):
        unprot_input, all_prot_input, target, _idx = dataset[:]

        prediction = predictor.predict(unprot_input)
        all_prot_input = all_prot_input.numpy()
        target = target.numpy()

        # Construct a dataframe from the subset values.
        df = pd.DataFrame(data=np.stack([prediction, target], axis=1),
                          columns=['preds', 'target'])
        df['target'] = df['target'].replace({0:'negative', 1:'positive'})
        all_scores = self.overall_scores(df, title_suffix)

        prot_col_pointer = 0
        for prot_name in dataset.prot_names:
            if prot_name in dataset.cat_val_to_str():
                prot_to_str_dict = dataset.cat_val_to_str()[prot_name]
                prot_input = all_prot_input[:, prot_col_pointer:prot_col_pointer + len(prot_to_str_dict)]
                df[prot_name] = pd.Series(np.argmax(prot_input, axis=1)).replace(prot_to_str_dict)
                all_prot_vals = list(prot_to_str_dict.values())
                cont_bin = None
            else:
                prot_input = all_prot_input[:, [prot_col_pointer]]
                df[prot_name] = prot_input
                all_prot_vals = ["continuous"]
                cont_bin = dataset.cont_bin()[prot_name]

            prot_title_suffix = f"{title_suffix}_{prot_name}"
            fairness_scores = self.fairness_scores(df, prot_input, prot_title_suffix, cont_bin)

            if log_plots:
                self.visualize(df, prot_name, all_prot_vals, prot_title_suffix)
                if unprot_input.shape[1] == 2:
                    unprot_input = unprot_input.numpy()
                    df['x0'] = unprot_input[:, 0]
                    df['x1'] = unprot_input[:, 1]
                    self.vis_decisions(df, predictor, prot_name, prot_title_suffix)

            prot_col_pointer += prot_input.shape[1]
            all_scores |= fairness_scores

        wandb.log(all_scores)
        return all_scores

    @staticmethod
    def overall_scores(df, title_suffix=''):
        scores = {
            f"roc_auc_{title_suffix}": skm.roc_auc_score(df['target'], df['preds'])
        }
        return scores

    @staticmethod
    def fairness_scores(df, prot_input, title_suffix, cont_bins):
        preds = df['preds']
        target = df['target']
        scores = {f"dp_{title_suffix}": Evaluator._pdp(preds, prot_input),
                  f"eo_{title_suffix}": Evaluator._peo(preds, prot_input, target)}

        # ROC-AUC for different groups.
        if prot_input.shape[1] > 1:
            aucs = [roc_auc_score(ovr_labels, preds, average='weighted') for ovr_labels in prot_input.T]
            scores[f"aucp_{title_suffix}"] = np.max(aucs)
        else:
            if cont_bins is None:
                print(f"No cont bins found for {title_suffix}!")
            else:
                prot_input_bin_idx = np.digitize(prot_input[:, 0], cont_bins)
                binned_prot_input = np.eye(len(cont_bins))[prot_input_bin_idx]
                new_title_suffix = f"{title_suffix}_binned"
                binned_scores = Evaluator.fairness_scores(df, binned_prot_input, new_title_suffix, None)
                scores |= binned_scores
        return scores

    @staticmethod
    def _pdp(preds, prot_input):
        all_corrs = np.corrcoef(preds, prot_input, ddof=0, rowvar=False)

        # Only interested in correlation between preds (first row) and sens vals (second+ cols).
        lin_ind = np.abs(all_corrs[0, 1:])
        max_lin_ind = np.max(lin_ind)
        return max_lin_ind

    @staticmethod
    def _peo(preds, prot_input, target):
        max_lin_ind_per_val = []
        for target_val in np.unique(target):
            target_val_idx = target == target_val
            preds_for_val = preds[target_val_idx]
            prot_input_for_val = prot_input[target_val_idx]
            max_lin_ind_for_val = Evaluator._pdp(preds_for_val, prot_input_for_val)
            max_lin_ind_per_val.append(max_lin_ind_for_val)
        max_lin_corr = np.max(max_lin_ind_per_val)
        return max_lin_corr

    @staticmethod
    def visualize(df, prot_name, all_prot_vals, title_suffix=''):
        df['all'] = 'all'
        for data_split in ['all', 'target']:
            if len(all_prot_vals) > 1:
                plt.figure(dpi=600)
                plt.ylim(0, 1)
                if len(df) > 10:
                    sns.violinplot(data=df, x=data_split, y='preds', hue=prot_name, hue_order=all_prot_vals,
                                   split=True, inner="quartile", bw=0.2)
                else:
                    sns.swarmplot(data=df, x=data_split, y='preds', hue=prot_name, hue_order=all_prot_vals,
                                  dodge=True)
            else:
                if data_split == 'all':
                    g = sns.lmplot(data=df, x=prot_name, y='preds')
                else:
                    g = sns.lmplot(data=df, x=prot_name, y='preds', hue=data_split, markers=["o", "x"])
                g.set(ylim=(0, 1))

            if data_split == 'all':
                log_name = f"preds_{title_suffix}"
            else:
                log_name = f"preds_by_target_{title_suffix}"
            wandb.log({log_name: wandb.Image(plt)})
            plt.close()

    @staticmethod
    def vis_decisions(df, predictor, prot_name, title_suffix=''):
        res = 0.1
        x_min, x_max = df['x0'].min() - res * 3, df['x0'].max() + res * 3
        y_min, y_max = df['x1'].min() - res * 3, df['x1'].max() + res * 3
        xx, yy = np.meshgrid(np.arange(x_min, x_max, res), np.arange(y_min, y_max, res))
        data = np.c_[xx.ravel(), yy.ravel()]
        preds = predictor.predict(torch.from_numpy(data).float())
        preds = preds.reshape(xx.shape)

        cmap = plt.cm.coolwarm
        # cmap = plt.cm.gist_gray
        plt.figure()
        plt.contourf(xx, yy, preds, alpha=0.4, cmap=cmap)
        # markers = ['_', '+']
        # for i, (t_val, t_df) in enumerate(df.groupby('target')):
        #     plt.scatter(t_df['x0'], t_df['x1'], c=t_df[prot_name], s=150,
        #                 cmap=cmap, marker=markers[i])
        markers = ['o', '^']
        df['c'] = df['target'].replace({'positive':1, 'negative':0})
        for prot_i, (prot_val, group_df) in enumerate(df.groupby(prot_name)):
            plt.scatter(group_df['x0'], group_df['x1'], c=group_df['c'], s=150, edgecolor="k",
                        cmap=cmap, marker=markers[prot_i])
            # for row_i, row in group_df.iterrows():
            #     x = row['x0']
            #     y = row['x1']
            #     plt.annotate(row_i, (x, y), xytext=(x+res/2, y+res/2))

        # ax = plt.gca()
        wandb.log({f"contour_{title_suffix}":wandb.Image(plt)})
        plt.show()
        pass
