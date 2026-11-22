
# src/model.py

import torch
import torch.nn as nn


# =====================================================
# CONFIGURATION
# =====================================================

class ModelConfig:

    INPUT_DIM = 12
    HIDDEN_DIM = 64
    NUM_LAYERS = 2
    OUTPUT_DIM = 1
    DROPOUT = 0.2


# =====================================================
# LSTM MODEL
# =====================================================

class LSTMModel(nn.Module):

    def __init__(
        self,
        input_dim=ModelConfig.INPUT_DIM,
        hidden_dim=ModelConfig.HIDDEN_DIM,
        num_layers=ModelConfig.NUM_LAYERS,
        output_dim=ModelConfig.OUTPUT_DIM,
        dropout=ModelConfig.DROPOUT
    ):

        super().__init__()

        self.input_dim = input_dim
        self.hidden_dim = hidden_dim
        self.num_layers = num_layers
        self.output_dim = output_dim

        # ---------------------------------------------
        # LSTM LAYERS
        # ---------------------------------------------

        self.lstm = nn.LSTM(
            input_size=input_dim,
            hidden_size=hidden_dim,
            num_layers=num_layers,
            batch_first=True,
            dropout=dropout if num_layers > 1 else 0.0
        )

        # ---------------------------------------------
        # FULLY CONNECTED OUTPUT LAYER
        # ---------------------------------------------

        self.fc = nn.Linear(
            hidden_dim,
            output_dim
        )

        # ---------------------------------------------
        # WEIGHT INITIALIZATION
        # ---------------------------------------------

        self._initialize_weights()

    def _initialize_weights(self):

        # Initialize fully connected layer.
        nn.init.xavier_uniform_(
            self.fc.weight
        )

        nn.init.zeros_(
            self.fc.bias
        )

    # ---------------------------------------------
    # FORWARD PASS
    # ---------------------------------------------

    def forward(self, x):

        """
        Input:
            x: (batch_size, sequence_length, input_dim)

        Output:
            predictions: (batch_size, output_dim)
        """

        if x.ndim != 3:
            raise ValueError(
                "Expected input shape "
                "(batch_size, sequence_length, input_dim)"
            )

        if x.shape[-1] != self.input_dim:
            raise ValueError(
                f"Expected {self.input_dim} input features, "
                f"received {x.shape[-1]}"
            )

        # LSTM processes the complete sequence.
        #
        # output shape:
        # (batch_size, sequence_length, hidden_dim)
        #
        # hidden shape:
        # (num_layers, batch_size, hidden_dim)
        #
        # cell shape:
        # (num_layers, batch_size, hidden_dim)

        output, (hidden, cell) = self.lstm(x)

        # Extract final time-step representation.
        #
        # Shape:
        # (batch_size, hidden_dim)

        last_output = output[:, -1, :]

        # Predict next-hour consumption.
        #
        # Shape:
        # (batch_size, output_dim)

        predictions = self.fc(last_output)

        return predictions


# =====================================================
# MODEL TEST
# =====================================================

if __name__ == "__main__":

    model = LSTMModel()

    print(model)

    # Simulate one training batch.
    batch_size = 64
    sequence_length = 24
    input_features = 12

    dummy_input = torch.randn(
        batch_size,
        sequence_length,
        input_features
    )

    # Forward pass.
    predictions = model(dummy_input)

    print("\nInput shape:")
    print(dummy_input.shape)

    print("\nOutput shape:")
    print(predictions.shape)

    # Parameter count.
    total_parameters = sum(
        parameter.numel()
        for parameter in model.parameters()
    )

    trainable_parameters = sum(
        parameter.numel()
        for parameter in model.parameters()
        if parameter.requires_grad
    )

    print("\nTotal parameters:", total_parameters)

    print(
        "Trainable parameters:",
        trainable_parameters
    )

    assert predictions.shape == (64, 1)

    print("\nModel test passed!")
