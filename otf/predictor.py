import torch
from tqdm import tqdm


class Predictor:
    def __init__(self, nb_features,
                 layer_sizes=None,
                 lr=.001,
                 nb_epochs=100,
                 fairness_loss_strength=0.5):
        super().__init__()
        self.lr = lr
        self.nb_epochs = nb_epochs
        self.batch_size = batch_size
        self.fairness_loss_strength = fairness_loss_strength

        self._model = None
        self._optimizer = None
        self._bce_loss_f = None
        self._fairness_loss_f = None

    def _build_model(self, feature_shape):
        modules = []
        last_nb_features = feature_shape
        for layer_size in self.layer_sizes:
            modules.append(torch.nn.Linear(in_features=last_nb_features, out_features=layer_size))
            modules.append(torch.nn.ReLU())
            last_nb_features = layer_size
        modules.append(torch.nn.Linear(in_features=last_nb_features, out_features=1))
        modules.append(torch.nn.Flatten(start_dim=0))
        model = torch.nn.Sequential(*modules)
        return model

    def fit(self, train_dataloader, fairness_cost):
        if not (0. <= self.fairness_loss_strength <= 1.):
            raise ValueError("Expected fairness strength to be between 0 and 1!")

        self._model = self._build_model(dataset.shape[1])
        self._optimizer = torch.optim.Adam(self._model.parameters(), lr=self.lr)
        self._bce_loss_f = torch.nn.BCEWithLogitsLoss()

        self._fairness_loss_f = fairness_cost

        for _ in tqdm(range(self.nb_epochs)):
            epoch_bce_loss = 0.
            epoch_fairness_loss = 0.
            for batch in train_dataloader:
                self._optimizer.zero_grad()

                unprot_input, prot_input, targets, C = batch
                logits = self._model(unprot_input)

                if fairness_strength < 1.:
                    bce_loss = self._bce_loss_f(logits, targets.float())
                else:
                    bce_loss = torch.zeros(1)
                epoch_bce_loss += bce_loss.item()

                if fairness_strength > 0.:
                    fairness_loss = self._fairness_loss_f(logits, prot_input, targets, C)
                else:
                    fairness_loss = torch.zeros(1)
                epoch_fairness_loss += fairness_loss.item()

                batch_loss = (1 - fairness_strength) * bce_loss + fairness_strength * fairness_loss
                batch_loss.backward()
                self._optimizer.step()

            epoch_logs = {'loss_bce': epoch_bce_loss / len(train_loader),
                          'loss_fairness': epoch_fairness_loss / len(train_loader)}
            if self._fairness_loss_f is not None:
                epoch_logs |= self._fairness_loss_f.get_epoch_logs()
            wandb.log(epoch_logs)

    def predict(self, unprot_input):
        self._model.eval()
        with torch.no_grad():
            logits = self._model(unprot_input)
            preds = torch.sigmoid(logits).detach().numpy()
            return preds
