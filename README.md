# Diffusion-Based Decoding for PAC Codes

This repository contains a research implementation of diffusion-based decoding for Polarization-Adjusted Convolutional (PAC) codes over an AWGN channel.

Current experiments include:

- PAC(16,8) and PAC(32,16)
- BPSK modulation and AWGN channel simulation
- LLR-based decoder input
- Conditional diffusion decoding
- FER and BER evaluation

## PAC Encoding

PAC encoding is implemented in `polar_codes/PAC_code.py`, including rate-profile placement, convolutional precoding, and Polar transformation.

The experiments currently use the `rm-polar` rate profile.

## Diffusion Decoder

The main decoder is a conditional diffusion model that reconstructs information bits from the received channel LLR vector.

The network contains:

- Channel-condition encoder
- Diffusion noise-prediction head
- Auxiliary information-prediction head
- Sinusoidal timestep embedding
- Residual fully connected blocks

Training uses a DDPM-style process with 1000 diffusion timesteps. Final evaluation uses 50 reverse-diffusion steps. Classifier-free guidance is used during training and inference.

## Experiments

### PAC(16,8)

- N = 16, K = 8
- Code rate = 0.5
- Hidden dimension = 256
- Residual blocks = 4
- Batch size = 4096
- Training epochs = 500

### PAC(32,16)

- N = 32, K = 16
- Code rate = 0.5
- Hidden dimension = 512
- Residual blocks = 6
- Batch size = 4096
- Training epochs = 800

Training samples are generated at Eb/N0 = 1, 2, 3, 4, 5 dB.

A CUDA GPU is recommended, especially for PAC(32,16), which requires substantially more memory and computation.

## Repository Structure

```text
.
├── DNN_16_8_dualhead.ipynb
├── DNN_32_16_dualhead.ipynb
├── MLP.path
├── polar_codes
│   ├── PAC_code.py
│   ├── rate_profile.py
│   ├── polar_coding_functions.py
│   ├── polar_coding_exceptions.py
│   └── channels
│       ├── channel.py
│       └── bpsk_awgn_channel.py
└── README.md
```

## Main Modules

- `polar_codes/PAC_code.py`: PAC encoding, rate-profile placement, convolutional precoding, and Polar transformation.
- `polar_codes/rate_profile.py`: Polar/PAC rate-profile construction.
- `polar_codes/polar_coding_functions.py`: Polar/PAC utilities, convolutional encoding, LLR operations, CRC, shortening, and puncturing.
- `polar_codes/channels/bpsk_awgn_channel.py`: BPSK modulation, AWGN transmission, demodulation, and LLR calculation.

## Dependencies

Main Python packages:

- numpy
- scipy
- torch
- scikit-learn
- diffusers
- matplotlib
- jupyter

Install with:

```bash
pip install numpy scipy torch scikit-learn diffusers matplotlib jupyterlab
```

Exact package versions are not currently fixed in the repository.

## Running

Clone the repository:

```bash
git clone https://github.com/glantbey1127/polarization-adjusted-convolutional-decode-based-on-diffusion-model.git
cd polarization-adjusted-convolutional-decode-based-on-diffusion-model
```

Start Jupyter:

```bash
jupyter lab
```

Run either:

```text
DNN_16_8_dualhead.ipynb
DNN_32_16_dualhead.ipynb
```

PAC(16,8) is recommended for a first test because it requires less memory and training time.

## Evaluation

The notebooks evaluate:

- Frame Error Rate (FER)
- Bit Error Rate (BER)
- Decoding time

Current evaluation configuration:

```text
Eb/N0 = 0 to 7 dB
Frames per Eb/N0 = 10000
Diffusion inference steps = 50
```

## Notes

This repository is currently research code.
