"""
Base Paper 1D-CNN Model Architecture
Reference: "Automatic Detection and Analysis of Singing Mistakes for Music Pedagogy"
Authors: Sumit Kumar, Suraj Jaiswal, Parampreet Singh, Vipul Arora (MADHAV Lab, IIT Kanpur)
Repository: madhavlab/2023_narottam_engine

Input:
    Multi-channel temporal tensor of shape (batch_size, sequence_length, 4)
    Channel 0: Student Pitch Contour (log2 scale relative to tonic)
    Channel 1: Teacher Reference Pitch Contour (log2 scale relative to tonic)
    Channel 2: Student Log Energy Contour
    Channel 3: Teacher Reference Log Energy Contour

Output:
    Tensor of shape (batch_size, sequence_length, 2) with sigmoid activations:
    Output 0: Probability of Frequency/Pitch Mistake ('F' - Flat / Apaswaram)
    Output 1: Probability of Amplitude/Dynamics Mistake ('A' - Volume inconsistency)
"""

try:
    import torch
    import torch.nn as nn
    TORCH_AVAILABLE = True
except ImportError:
    TORCH_AVAILABLE = False


if TORCH_AVAILABLE:
    class BasePaperCNN(nn.Module):
        """
        PyTorch implementation of the 10-layer 1D-CNN from the Base Paper.
        """
        def __init__(self, in_channels: int = 4, out_channels: int = 2):
            super().__init__()
            # Filters: 32 -> 64 -> 128 -> 256 -> 256 -> 128 -> 128 -> 64 -> 32 -> 16 -> 2
            self.net = nn.Sequential(
                # Block 1
                nn.Conv1d(in_channels, 32, kernel_size=3, padding=1, bias=False),
                nn.BatchNorm1d(32),
                nn.ReLU(inplace=True),

                # Block 2
                nn.Conv1d(32, 64, kernel_size=3, padding=1, bias=False),
                nn.BatchNorm1d(64),
                nn.ReLU(inplace=True),

                # Block 3
                nn.Conv1d(64, 128, kernel_size=3, padding=1, bias=False),
                nn.BatchNorm1d(128),
                nn.ReLU(inplace=True),

                # Block 4
                nn.Conv1d(128, 256, kernel_size=3, padding=1, bias=False),
                nn.BatchNorm1d(256),
                nn.ReLU(inplace=True),

                # Block 5
                nn.Conv1d(256, 256, kernel_size=3, padding=1, bias=False),
                nn.BatchNorm1d(256),
                nn.ReLU(inplace=True),

                # Block 6
                nn.Conv1d(256, 128, kernel_size=3, padding=1, bias=False),
                nn.BatchNorm1d(128),
                nn.ReLU(inplace=True),

                # Block 7
                nn.Conv1d(128, 128, kernel_size=3, padding=1, bias=False),
                nn.BatchNorm1d(128),
                nn.ReLU(inplace=True),

                # Block 8
                nn.Conv1d(128, 64, kernel_size=3, padding=1, bias=False),
                nn.BatchNorm1d(64),
                nn.ReLU(inplace=True),

                # Block 9
                nn.Conv1d(64, 32, kernel_size=3, padding=1, bias=False),
                nn.BatchNorm1d(32),
                nn.ReLU(inplace=True),

                # Block 10
                nn.Conv1d(32, 16, kernel_size=3, padding=1, bias=False),
                nn.BatchNorm1d(16),
                nn.ReLU(inplace=True),

                # Output Conv
                nn.Conv1d(16, out_channels, kernel_size=3, padding=1, bias=True),
                nn.Sigmoid()
            )

        def forward(self, x: torch.Tensor) -> torch.Tensor:
            # x shape: (batch, seq_len, channels) -> transpose to (batch, channels, seq_len)
            if x.dim() == 3 and x.shape[1] != 4 and x.shape[2] == 4:
                x = x.transpose(1, 2)
            out = self.net(x)
            # return shape: (batch, seq_len, 2)
            return out.transpose(1, 2)


def build_keras_base_paper_cnn(seq_len: int = None, in_channels: int = 4, out_channels: int = 2):
    """
    Builds the original Keras/TensorFlow Sequential Model matching model_001.h5 / model_002.h5.
    """
    try:
        from tensorflow import keras
        from tensorflow.keras.layers import Conv1D, BatchNormalization
    except ImportError:
        import keras
        from keras.layers import Conv1D, BatchNormalization

    model = keras.models.Sequential([
        Conv1D(filters=32, kernel_size=3, activation='relu', input_shape=(seq_len, in_channels), padding='same', use_bias=False),
        BatchNormalization(),
        Conv1D(filters=64, kernel_size=3, activation='relu', padding='same', use_bias=False),
        BatchNormalization(),
        Conv1D(filters=128, kernel_size=3, activation='relu', padding='same', use_bias=False),
        BatchNormalization(),
        Conv1D(filters=256, kernel_size=3, activation='relu', padding='same', use_bias=False),
        BatchNormalization(),
        Conv1D(filters=256, kernel_size=3, activation='relu', padding='same', use_bias=False),
        BatchNormalization(),
        Conv1D(filters=128, kernel_size=3, activation='relu', padding='same', use_bias=False),
        BatchNormalization(),
        Conv1D(filters=128, kernel_size=3, activation='relu', padding='same', use_bias=False),
        BatchNormalization(),
        Conv1D(filters=64, kernel_size=3, activation='relu', padding='same', use_bias=False),
        BatchNormalization(),
        Conv1D(filters=32, kernel_size=3, activation='relu', padding='same', use_bias=False),
        BatchNormalization(),
        Conv1D(filters=16, kernel_size=3, activation='relu', padding='same', use_bias=False),
        BatchNormalization(),
        Conv1D(filters=out_channels, kernel_size=3, activation='sigmoid', padding='same', use_bias=True)
    ])
    model.compile(loss='binary_crossentropy', optimizer='adam', metrics=['accuracy'])
    return model


if __name__ == "__main__":
    print("=" * 65)
    print("[*] BASE PAPER 1D-CNN ARCHITECTURE SPECIFICATION")
    print("=" * 65)
    print("Architecture: 10-layer Deep 1D Convolutional Neural Network")
    print("Channels in : 4 (Student Pitch, Teacher Pitch, Student RMS, Teacher RMS)")
    print("Channels out: 2 (Mistake_Frequency 'F', Mistake_Amplitude 'A')")
    print("Activations : ReLU for hidden layers, Sigmoid for frame probabilities")
    print("Regularizer : Batch Normalization after every Conv1D layer")
    print("Loss Func   : Binary Cross-Entropy")
    print("=" * 65)
