import torch
from torch import nn, optim
from torch.nn import functional as F


class ModelWithTemperature(nn.Module):
    """
    A thin decorator, which wraps a model with temperature scaling
    model (nn.Module):
        A classification neural network
        NB: Output of the neural network should be the classification logits,
            NOT the softmax (or log softmax)!
    """
    def __init__(self, model, device):
        super(ModelWithTemperature, self).__init__()
        self.device = device
        self.model = model
        self.temperature = torch.ones(1)
        self.temperature = self.temperature.to(self.device)
        self.temperature.requires_grad = True
        self.temperature = nn.Parameter(self.temperature)


# forward所输出的是经过调整的logits
    def forward(self, input):
        logits = self.model(input)
        return self.temperature_scale(logits)

    def temperature_scale(self, logits):
        """
        Perform temperature scaling on logits
        """
        # Expand temperature to match the size of logits
        logits = logits.to(self.device)
        temperature = self.temperature.unsqueeze(1).expand(logits.size(0), logits.size(1)).to(self.device)
        return logits / temperature

    def predict(self, X):
        if not isinstance(X, torch.Tensor):
            X = torch.tensor(X)
        self.model = self.model.to(self.device)
        outputs = self.forward(X)
        _, predicted = torch.max(outputs.data, 1)
        return predicted.detach().cpu().numpy()

    def predict_proba(self, X):
        if not isinstance(X, torch.Tensor):
            X = torch.tensor(X)
        inputs = X.to(self.device)
        outputs = self.forward(inputs)
        # print(outputs.shape)
        softmax = nn.Softmax(dim=1)
        outputs = softmax(outputs)
        return outputs.detach().cpu().numpy()

# score函数是根据logits来得到模型预测结果的，而T变换实际上并不会影响logits之间的相互关系，所以此函数中是否通过T变换没有影响
    def score(self, X, y):
        if not isinstance(X, torch.Tensor):
            X = torch.tensor(X)
        if not isinstance(y, torch.Tensor):
            y = torch.tensor(y)
        inputs, labels = X.to(self.device), y.to(self.device)
        self.model = self.model.to(self.device)
        outputs = self.forward(inputs)
        _, predicted = torch.max(outputs.data, 1)
        correct = (predicted == labels).sum().item()
        total = y.shape[0]
        return correct / total

    # This function probably should live outside of this class, but whatever
    def set_temperature(self, valid_loader):
        """
        Tune the tempearature of the model (using the validation set).
        We're going to set it to optimize NLL.
        valid_loader (DataLoader): validation set loader
        """
        self.to(self.device)
        nll_criterion = nn.CrossEntropyLoss().to(self.device)
        ece_criterion = _ECELoss().to(self.device)

        # First: collect all the logits and labels for the validation set
        logits_list = []
        labels_list = []
        with torch.no_grad():
            for i, data in enumerate(valid_loader, 0):
                inputs, labels, *weights_batch = data
                labels = labels.long()
                inputs = inputs.to(self.device)
                logits = self.model(inputs)
                logits_list.append(logits)
                labels_list.append(labels)
            logits = torch.cat(logits_list).to(self.device)
            labels = torch.cat(labels_list).to(self.device)

        # Calculate NLL and ECE before temperature scaling
        before_temperature_nll = nll_criterion(logits, labels).item()
        before_temperature_ece = ece_criterion(logits, labels).item()
        print('Before temperature - NLL: %.3f, ECE: %.3f' % (before_temperature_nll, before_temperature_ece))

        # Next: optimize the temperature w.r.t. NLL
        optimizer = optim.LBFGS([self.temperature], lr=0.1, max_iter=50)

        def eval():
            optimizer.zero_grad()
            loss = nll_criterion(self.temperature_scale(logits).to(self.device), labels.to(self.device))
            # print(loss.item())
            loss.backward()
            return loss
        optimizer.step(eval)

        # Calculate NLL and ECE after temperature scaling
        after_temperature_nll = nll_criterion(self.temperature_scale(logits).to(self.device), labels.to(self.device)).item()
        after_temperature_ece = ece_criterion(self.temperature_scale(logits).to(self.device), labels.to(self.device)).item()
        print('Optimal temperature: %.3f' % self.temperature.item())
        print('After temperature - NLL: %.3f, ECE: %.3f' % (after_temperature_nll, after_temperature_ece))

        return self


class _ECELoss(nn.Module):
    """
    Calculates the Expected Calibration Error of a model.
    (This isn't necessary for temperature scaling, just a cool metric).

    The input to this loss is the logits of a model, NOT the softmax scores.

    This divides the confidence outputs into equally-sized interval bins.
    In each bin, we compute the confidence gap:

    bin_gap = | avg_confidence_in_bin - accuracy_in_bin |

    We then return a weighted average of the gaps, based on the number
    of samples in each bin

    See: Naeini, Mahdi Pakdaman, Gregory F. Cooper, and Milos Hauskrecht.
    "Obtaining Well Calibrated Probabilities Using Bayesian Binning." AAAI.
    2015.
    """
    def __init__(self, n_bins=15):
        """
        n_bins (int): number of confidence interval bins
        """
        super(_ECELoss, self).__init__()
        bin_boundaries = torch.linspace(0, 1, n_bins + 1)
        self.bin_lowers = bin_boundaries[:-1]
        self.bin_uppers = bin_boundaries[1:]

    def forward(self, logits, labels):
        softmaxes = F.softmax(logits, dim=1)
        confidences, predictions = torch.max(softmaxes, 1)
        accuracies = predictions.eq(labels)

        ece = torch.zeros(1, device=logits.device)
        for bin_lower, bin_upper in zip(self.bin_lowers, self.bin_uppers):
            # Calculated |confidence - accuracy| in each bin
            in_bin = confidences.gt(bin_lower.item()) * confidences.le(bin_upper.item())
            prop_in_bin = in_bin.float().mean()
            if prop_in_bin.item() > 0:
                accuracy_in_bin = accuracies[in_bin].float().mean()
                avg_confidence_in_bin = confidences[in_bin].mean()
                ece += torch.abs(avg_confidence_in_bin - accuracy_in_bin) * prop_in_bin

        return ece