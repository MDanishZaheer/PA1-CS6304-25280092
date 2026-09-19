# Name : Muhammad Danish Zaheer Awan
# REG id : 25280092

"""Define the binary domain discriminator shared by DANN and CDAN."""

from torch import nn


class DomainDiscriminator(nn.Module):
    """Predict source or target domain from an adaptation representation."""

    def __init__(self, input_dimension, hidden_dimension=256, dropout=0.5):
        super().__init__()
        if input_dimension < 1 or hidden_dimension < 1:
            raise ValueError("Discriminator dimensions must be positive.")
        if not 0.0 <= dropout < 1.0:
            raise ValueError("Discriminator dropout must lie in [0, 1).")

        self.network = nn.Sequential(
            nn.Linear(input_dimension, hidden_dimension),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dimension, 2),
        )

    def forward(self, representations):
        return self.network(representations)


def build_domain_discriminator(configuration, device):
    """Build the correctly sized discriminator for DANN or CDAN."""
    method = configuration["method"]
    method_name = method["name"]
    if method_name == "dann":
        input_dimension = configuration["model"]["feature_dimension"]
    elif method_name == "cdan":
        input_dimension = method["discriminator_input_dimension"]
        expected_dimension = (
            configuration["model"]["feature_dimension"]
            * configuration["model"]["number_of_classes"]
        )
        if input_dimension != expected_dimension:
            raise ValueError("CDAN discriminator input must equal 512 times 7.")
    else:
        return None

    discriminator = DomainDiscriminator(
        input_dimension=input_dimension,
        hidden_dimension=method["discriminator_hidden_dimension"],
        dropout=method["discriminator_dropout"],
    )
    return discriminator.to(device)
