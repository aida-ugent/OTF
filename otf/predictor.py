import torch
from tqdm import tqdm


class Predictor:
    def __init__(self, nb_features: int,
                 layer_sizes: list[int] = None,
                 lr: float = .001,
                 nb_epochs: int = 100,
                 fairness_loss_strength: float = 0.5):
        """
        Generic neural network model that optimizes the convex sum of a binary cross-entropy loss and a fairness
        regularization term.
        :param nb_features: number of input features in each feature vector.
        :param layer_sizes: dimensionality of hidden layers. Can be empty.
        :param lr: learning rate.
        :param nb_epochs: number of training epochs.
        :param fairness_loss_strength: (aka alpha) relative strength of the fairness regularization loss over the binary
        cross-entropy loss on a scale of 0 to 1. For efficiency reasons, the fairness loss is not computed for value 0.
        """
        super().__init__()
        self.nb_features = nb_features
        if layer_sizes is None:
            layer_sizes = []
        self.layer_sizes = layer_sizes
        self.lr = lr
        self.nb_epochs = nb_epochs
        self.fairness_loss_strength = fairness_loss_strength

        self.model = None
        self.optimizer = None
        self.bce_loss_f = None
        self.fairness_loss_f = None

    def _build_model(self):
        modules = []
        last_nb_features = self.nb_features
        for layer_size in self.layer_sizes:
            modules.append(torch.nn.Linear(in_features=last_nb_features, out_features=layer_size))
            modules.append(torch.nn.ReLU())
            last_nb_features = layer_size
        modules.append(torch.nn.Linear(in_features=last_nb_features, out_features=1))
        modules.append(torch.nn.Flatten(start_dim=0))
        model = torch.nn.Sequential(*modules)
        return model

    def fit(self, train_dataloader, fairness_cost_f):
        if not (0. <= self.fairness_loss_strength <= 1.):
            raise ValueError("Expected fairness strength to be between 0 and 1!")

        self.model = self._build_model()
        self.optimizer = torch.optim.Adam(self.model.parameters(), lr=self.lr)
        self.bce_loss_f = torch.nn.BCEWithLogitsLoss()
        self.fairness_loss_f = fairness_cost_f

        for _ in tqdm(range(self.nb_epochs)):
            for batch in train_dataloader:
                self.optimizer.zero_grad()

                unprot_input, prot_input, targets, *extra = batch
                logits = self.model(unprot_input)

                if self.fairness_loss_strength < 1.:
                    bce_loss = self.bce_loss_f(logits, targets.float())
                else:
                    bce_loss = torch.zeros(1)

                if self.fairness_loss_strength > 0.:
                    fairness_loss = self.fairness_loss_f(logits, prot_input, targets, *extra)
                else:
                    fairness_loss = torch.zeros(1)

                batch_loss = (1 - self.fairness_loss_strength) * bce_loss + self.fairness_loss_strength * fairness_loss
                batch_loss.backward()
                self.optimizer.step()

    def predict(self, unprot_input):
        self.model.eval()
        with torch.no_grad():
            logits = self.model(unprot_input)
            preds = torch.sigmoid(logits)
            return preds
