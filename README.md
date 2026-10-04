# Diffusion-Based Decoding for PAC Codes
This repository contains a research implementation of diffusion-based decoding for Polarization-Adjusted Convolutional (PAC) codes over an AWGN channel.
The current experiments include:
- PAC(16,8) and PAC(32,16)
- BPSK modulation
- AWGN channel simulation
- LLR-based decoder input
- Conditional diffusion decoding
- FER and BER evaluation

## PAC Encoding
The PAC encoding process is implemented in `polar_codes/PAC_code.py`.

## Diffusion Decoder
The main decoder is a conditional diffusion model.
The decoder takes the received channel LLR vector as condition information and reconstructs the original information bits.
The network contains:
- a channel-condition encoder
- a diffusion noise-prediction head
- an auxiliary information-prediction head
- sinusoidal timestep embeddings
- residual fully connected blocks
Training uses a DDPM-style noise process with 1000 diffusion timesteps.The final simulations use 50 reverse-diffusion steps.Classifier-free guidance is also used during training and inference.

## Experiments
Main settings:
-N = 16,K = 8,Code rate = 0.5
-Hidden dimension = 256
-Residual blocks = 4
-Batch size = 4096
-Training epochs = 500
Training samples are generated at:Eb/N0 = 1, 2, 3, 4, 5 dB
Because this experiment requires significantly more memory and computation, a CUDA GPU is recommended.

## Repository Structure
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

## Main Modules
polar_codes/PAC_code.py:Implements PAC encoding, rate-profile placement, convolutional precoding and Polar transformation.
polar_codes/rate_profile.py:Implements Polar/PAC rate-profile construction, including RM-Polar, Bhattacharyya, DEGA and polarization-weight based methods.
polar_codes/polar_coding_functions.py:Contains Polar and PAC utility functions, convolutional encoding, LLR operations, CRC functions, shortening and puncturing utilities.
polar_codes/channels/bpsk_awgn_channel.py:Implements BPSK modulation, AWGN transmission, hard demodulation and LLR calculation.

## Dependencies
Main Python packages:
-numpy
-scipy
-torch
-scikit-learn
-diffusers
-matplotlib
-jupyter

Install them with:
pip install numpy scipy torch scikit-learn diffusers matplotlib jupyterlab

Exact package versions are not currently fixed in the repository.

## Running
Clone the repository:
```bash
git clone https://github.com/glantbey1127/polarization-adjusted-convolutional-decode-based-on-diffusion-model.git
```

Enter the directory:
```bash
cd polarization-adjusted-convolutional-decode-based-on-diffusion-model
```

Start Jupyter:
```bash
jupyter lab
```

Then run either:
```text
DNN_16_8_dualhead.ipynb
```
or:
```text
DNN_32_16_dualhead.ipynb
```
The PAC(16,8) experiment is recommended for the first test because it requires less memory and training time.

## Evaluation
The notebooks evaluate:
- Frame Error Rate (FER)
- Bit Error Rate (BER)
- decoding time

over multiple Eb/N0 values.
The current simulation configuration uses:
```text
Eb/N0 = 0 to 7 dB
Frames per Eb/N0 = 10000
Diffusion inference steps = 50
```

## Notes
This repository is currently research code.
