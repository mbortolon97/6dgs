import math

import torch


def roots(coeffs):
    # Check if the input is not empty and a vector
    if math.prod(coeffs.shape) == 0:
        raise ValueError("Input must be a non-empty vector")

    # Check for non-finite elements in the input
    finite_coefficients = torch.isfinite(coeffs).all(dim=-1)

    breakpoint()
    # Find the first non-zero element
    inz = torch.nonzero(coeffs)
    inz[:, 0] * 3 + inz[:, 1]
    if torch.numel(inz) == 0:
        # All elements are zero
        return torch.tensor([], device=coeffs.device, dtype=coeffs.dtype)

    # Strip leading zeros and throw them away
    coeffs = coeffs[..., inz[0]]

    # Prevent relatively small leading coefficients from introducing Inf
    d = coeffs[..., 1:] / coeffs[..., 0]
    while torch.isinf(d).any():
        coeffs = coeffs[1:]
        d = coeffs[1:] / coeffs[0]

    n = coeffs.numel()

    if n > 1:
        # Create the companion matrix
        a = torch.zeros((n - 1, n - 1), device=coeffs.device, dtype=coeffs.dtype)
        a[0, :] = -d
        a[1 :: n + 1] = 1

        # Compute eigenvalues of the companion matrix
        eigvals = torch.linalg.eigvals(a)[:, 0]

        # Add any roots at zero
        r = torch.cat(
            (
                eigvals,
                torch.zeros(
                    n - eigvals.numel(), device=coeffs.device, dtype=coeffs.dtype
                ),
            )
        )

        return r

    # If n == 1, the polynomial is just a constant
    return torch.tensor([], device=coeffs.device, dtype=coeffs.dtype)
