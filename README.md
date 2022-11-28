# Optimal Transport to Fairness (OTF)

This project contains an accessible implementation of the OTF cost function proposed in the paper *Optimal Transport of Classifiers to Fairness* published at NeurIPS 2022. 

The OTF cost projects a probability distribution to the closest distribution in the set of all fair distributions, where closeness is defined in terms of Optimal Transport cost. As such, OTF quantifies the unfairness of a model while taking the input features of individuals into account.


### Use

An example use of the OTF method for the Adult dataset is given in `main.py` . The actual implementation of the cost is in `otf.otf_cost`, where a fairness notion as given in `otf.linear_fairness_notion` is expected. In `otf.predictor`, a generic probabilistic model is implemented that uses the OTF cost as an additional cost term to optimize during training. Finally, `otf.evaluation` computes some metrics as explained in the paper.

When using the OTF cost, please make sure to tune the `reg_strength` hyperparameter at the very least, e.g. in the range `[0.01, 0.001, 0.0001]`. Also, the computation of the OTF cost can be sped up by reducing the `nb_epochs` parameter and increasing the `margin_tol` and `constraint_tol`. 


### Citation

If you found our code useful, please cite our paper:


    @inproceedings{buyl2022otf,
        title={Optimal Transport of Classifiers to Fairness},
        author={Buyl, Maarten and De Bie, Tijl},
        booktitle={Conference on Neural Information Processing Systems}
    }

Note: though the paper was presented at NeurIPS 2022, the proceedings were not published yet. In the meantime, you can also check out our arxiv version: https://arxiv.org/abs/2202.03814. 
