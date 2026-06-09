"""LIMA: lightweight image matching architecture for PIV.

Re-implementation of the lightweight optical-flow CNN of Manickathan,
Mucignat & Lunati (Exp. Fluids 2023) with the padding and search-range
improvements of Mucignat, Zdybał & Lunati (Phys. Fluids 2025).

References:
    [1] Manickathan, L., Mucignat, C., & Lunati, I. (2023). "A lightweight
        neural network designed for fluid velocimetry." Experiments in
        Fluids, 64, 161. https://doi.org/10.1007/s00348-023-03695-8
        (Defines the LIMA architecture and the multi-level Jacobian-penalised
        loss; the LIMA-6/LIMA-4 variants.)
    [2] Mucignat, C., Zdybał, K., & Lunati, I. (2025). "Improving the
        performance of a lightweight convolutional neural network for particle
        image velocimetry through hyper-parameter and padding optimization."
        Physics of Fluids, 37, 105112. https://doi.org/10.1063/5.0283779
        (Encoder/decoder layer tables, the zero/replicate padding study
        LIMA0/LIMAR, and the correlation search-range study.)
    [3] Manickathan, L., Mucignat, C., & Lunati, I. (2022). "Kinematic
        training of convolutional neural networks for particle image
        velocimetry." Measurement Science and Technology, 33, 124006.
        https://doi.org/10.1088/1361-6501/ac8fae (The kinematic training
        strategy used here: render images from random displacement fields.)
    [4] Hur, J., & Roth, S. (2019). "Iterative residual refinement for joint
        optical flow and occlusion estimation." CVPR.
        https://doi.org/10.1109/CVPR.2019.00590 (The weight-shared iterative
        residual refinement that LIMA builds on.)
    [5] Sun, D., Yang, X., Liu, M.-Y., & Kautz, J. (2018). "PWC-Net: CNNs for
        optical flow using pyramid, warping, and cost volume." CVPR.
        arXiv:1709.02371 (Pyramid + warping + cost-volume backbone.)
    [6] Wereley, S. T., & Meinhart, C. D. (2001). "Second-order accurate
        particle image velocimetry." Experiments in Fluids, 31, 258-268.
        https://doi.org/10.1007/s003480100281 (The symmetric, central-
        difference warping toward the temporal midpoint.)
"""

from flowgym.flow.lima.lima_piv import LimaPivEstimator

__all__ = ["LimaPivEstimator"]
