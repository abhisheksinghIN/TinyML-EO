# Quantised Deep Learning Model for Eearth Observation Application
The proposed Quantized-UNet can be interpreted as a discrete approximation of a standard UNet, where all continuous-valued feature transformations are projected onto a finite set of representable levels defined by the quantisation bit-width. Instead of operating in a continuous floating-point space, the network learns a mapping in a constrained integer lattice, where both weights and activations belong to uniformly quantised sets.

From a signal processing perspective, each layer performs a composition of three operations: (i) linear filtering via quantised convolution, (ii) distribution normalisation via batch normalisation, and (iii) nonlinear projection via quantised activation. This results in a piecewise-constant approximation of the underlying feature space, where representational capacity is controlled by the quantisation resolution.

Despite the reduced numerical precision, the encoder–decoder structure preserves multi-scale feature extraction and reconstruction. The encoder progressively compresses spatial information into a compact latent representation, the bottleneck captures global contextual dependencies under quantisation constraints, and the decoder reconstructs spatial detail through skip-connected feature fusion. This design enables efficient dense prediction while maintaining robustness to quantisation-induced noise.

Quantised models offer:
\begin{itemize}
    \item reduced memory footprint (8-bit weights vs.\ 32-bit floating point),
    \item reduced compute cost (integer MACs),
    \item reduced energy consumption,
    \item improved suitability for FPGA/ASIC acceleration,
    \item robustness due to flatter loss landscapes.
\end{itemize}
