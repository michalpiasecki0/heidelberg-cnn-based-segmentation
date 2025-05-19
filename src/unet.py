from torch import nn
import torch


def weights_init(m):
    if isinstance(m, nn.Conv2d):
        nn.init.kaiming_normal_(m.weight, mode="fan_in", nonlinearity="relu")
    if isinstance(m, nn.ConvTranspose2d):
        nn.init.kaiming_normal_(m.weight, mode="fan_in", nonlinearity="relu")


class EncoderLayer(nn.Module):
    def __init__(self, ch_in, ch_out, kernel_size, padding, pooling, dropout):
        super(EncoderLayer, self).__init__()
        if pooling:
            self.pooling = nn.MaxPool2d(2)
        else:
            self.pooling = None

        self.block = nn.Sequential(
            nn.Conv2d(
                ch_in, ch_out, kernel_size=kernel_size, stride=1, padding=padding
            ),
            nn.ReLU(),
            nn.Dropout(dropout) if dropout > 0 else nn.Identity(),
            nn.Conv2d(
                ch_out, ch_out, kernel_size=kernel_size, stride=1, padding=padding
            ),
            nn.ReLU(),
            nn.Dropout(dropout) if dropout > 0 else nn.Identity(),
        )

    def forward(self, x):
        if self.pooling is not None:
            x = self.pooling(x)
        x = self.block(x)
        return x


class EncoderLayerBN(EncoderLayer):
    def __init__(self, ch_in, ch_out, kernel_size, padding, pooling, dropout):
        super(EncoderLayerBN, self).__init__(
            ch_in, ch_out, kernel_size, padding, pooling, dropout
        )

        # your code goes here
        self.block = nn.Sequential(
            nn.Conv2d(ch_in, ch_out, kernel_size=kernel_size, padding=padding),
            nn.BatchNorm2d(ch_out),
            nn.ReLU(inplace=True),
            nn.Conv2d(ch_out, ch_out, kernel_size=kernel_size, padding=padding),
            nn.BatchNorm2d(ch_out),
            nn.ReLU(inplace=True),
        )


class DecoderLayer(nn.Module):
    def __init__(
        self,
        ch_in,
        ch_out,
        kernel_size,
        padding,
        dropout,
        skip_mode="concat",
        upsampling_mode="transpose",
        cropping=False,
    ):
        super(DecoderLayer, self).__init__()

        assert upsampling_mode in [
            "transpose",
            "interpolate",
        ], f"Upsampling has to be either 'transpose' or 'interpolate' but got '{upsampling_mode}'"
        assert skip_mode in [
            "concat",
            "add",
            "none",
        ], f"Skip-connection has to be either 'none', 'add' or 'concat' but got '{skip_mode}'"

        self.cropping = cropping
        self.skip_mode = skip_mode
        self.upsampling_mode = upsampling_mode

        if self.upsampling_mode == "transpose":
            self.up = nn.ConvTranspose2d(ch_in, ch_out, kernel_size=2, stride=2)
        else:
            self.up = nn.Upsample(scale_factor=2, mode="bilinear", align_corners=True)
            self.conv = nn.Conv2d(ch_in, ch_out, kernel_size=1, stride=1, padding=0)

        if self.skip_mode == "concat":
            ch_hidden = ch_out + ch_out
        elif self.skip_mode == "add" or self.skip_mode == "none":
            ch_hidden = ch_out

        self.block = nn.Sequential(
            nn.Conv2d(
                ch_hidden, ch_out, kernel_size=kernel_size, stride=1, padding=padding
            ),
            nn.ReLU(),
            nn.Dropout(dropout) if dropout > 0 else nn.Identity(),
            nn.Conv2d(
                ch_out, ch_out, kernel_size=kernel_size, stride=1, padding=padding
            ),
            nn.ReLU(),
            nn.Dropout(dropout) if dropout > 0 else nn.Identity(),
        )

    def crop(self, x, cropping_size):
        return x[
            :,
            :,
            cropping_size[0] : -cropping_size[0],
            cropping_size[1] : -cropping_size[1],
        ]

    def forward(self, x, skip_features):
        if self.upsampling_mode == "transpose":
            x = self.up(x)
        elif self.upsampling_mode == "interpolate":
            x = self.up(x)
            x = self.conv(x)

        if self.cropping:
            cropping_size = (
                torch.tensor(skip_features.shape[2:]) - torch.tensor(x.shape[2:])
            ) // 2
            skip_features = self.crop(skip_features, cropping_size)

        if self.skip_mode == "concat":
            x = self.block(torch.cat((x, skip_features), 1))
        elif self.skip_mode == "add":
            x = self.block(x + skip_features)
        elif self.skip_mode == "none":
            x = self.block(x)

        return x


class DecoderLayerBN(DecoderLayer):
    def __init__(
        self,
        ch_in,
        ch_out,
        kernel_size,
        padding,
        dropout,
        skip_mode="concat",
        upsampling_mode="transpose",
        cropping=False,
    ):
        super(DecoderLayerBN, self).__init__(
            ch_in,
            ch_out,
            kernel_size,
            padding,
            dropout,
            skip_mode,
            upsampling_mode,
            cropping,
        )

        if self.skip_mode == "concat":
            ch_hidden = ch_out + ch_out
        elif self.skip_mode == "add" or self.skip_mode == "add":
            ch_hidden = ch_out

        # your code goes here
        self.block = nn.Sequential(
            nn.Conv2d(ch_hidden, ch_out, kernel_size=kernel_size, padding=padding),
            nn.BatchNorm2d(ch_out),
            nn.ReLU(inplace=True),
            nn.Conv2d(ch_out, ch_out, kernel_size=kernel_size, padding=padding),
            nn.BatchNorm2d(ch_out),
            nn.ReLU(inplace=True),
        )


class UNet2d(nn.Module):
    def __init__(
        self,
        input_dim,
        output_dim,
        encoder_layer=EncoderLayer,
        decoder_layer=DecoderLayer,
        hidden_dims=[64, 128, 256, 512, 1024],
        kernel_size=3,
        padding_mode="valid",
        skip_mode="concat",
        upsampling_mode="transpose",
        dropout=0,
    ):

        super(UNet2d, self).__init__()

        assert len(hidden_dims) > 0, "UNet2d requires at least one hidden layer"
        assert padding_mode in [
            "same",
            "valid",
        ], f"Padding mode has to be either 'same' or 'valid' but got '{padding_mode}'"

        self.padding_mode = padding_mode

        cropping = True if padding_mode == "valid" else False
        padding = 0 if padding_mode == "valid" else kernel_size // 2

        # Assembling the encoder
        encoder = []

        # Add first encoder layer
        ch_in = input_dim
        ch_out = hidden_dims[0]
        encoder.append(
            encoder_layer(
                ch_in,
                ch_out,
                kernel_size=kernel_size,
                padding=padding,
                pooling=False,
                dropout=0,
            )
        )

        # Add intermediate encoder layers
        for i in range(1, len(hidden_dims) - 1):
            ch_in = hidden_dims[i - 1]
            ch_out = hidden_dims[i]
            encoder.append(
                encoder_layer(
                    ch_in,
                    ch_out,
                    kernel_size=kernel_size,
                    padding=padding,
                    pooling=True,
                    dropout=dropout,
                )
            )

        # Add last encoder layer
        ch_in = hidden_dims[i - 1]
        ch_out = hidden_dims[i]
        encoder.append(
            encoder_layer(
                ch_in,
                ch_out,
                kernel_size=kernel_size,
                padding=padding,
                pooling=True,
                dropout=0,
            )
        )
        self.encoder = nn.ModuleList(encoder)

        # Assembling the decoder
        decoder = []

        # Reversing the order of the hidden dims, since the decoder reduces the number of channels
        hidden_dims_rev = hidden_dims[::-1]

        for i in range(len(hidden_dims_rev) - 1):
            ch_in = hidden_dims_rev[i]
            ch_out = hidden_dims_rev[i + 1]
            decoder.append(
                decoder_layer(
                    ch_in,
                    ch_out,
                    kernel_size=kernel_size,
                    padding=padding,
                    dropout=0,
                    skip_mode=skip_mode,
                    upsampling_mode=upsampling_mode,
                    cropping=cropping,
                )
            )
        self.decoder = nn.ModuleList(decoder)

        # Creating final 1x1 convolution
        self.final_conv = nn.Conv2d(
            hidden_dims[0], output_dim, kernel_size=1, stride=1, padding=0
        )

    def forward(self, x):
        # Forward pass of the encoder
        skip_features = []
        for encoder_layer in self.encoder:
            x = encoder_layer(x)
            skip_features.insert(0, x)

        # Removing bottleneck features from the feature list
        skip_features = skip_features[1:]

        # Forward pass of the decoder
        for i, decoder_layer in enumerate(self.decoder):
            skip = skip_features[i]
            x = decoder_layer(x, skip)

        # Performing the final 1x1 convolution
        x = self.final_conv(x)
        return x
