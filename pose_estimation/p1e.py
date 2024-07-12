import numpy as np
import torch

from pose_estimation.roots import roots
from scene import GaussianModel
from scene.ellipsis_model import EllipsisModel
from xitorch.optimize import rootfinder


def compute_camera_position_solutions(
    ellipsoid: GaussianModel, B_: torch.Tensor, s: torch.Tensor
):
    Aw = ellipsoid.get_a_mat()[0]
    Cw = ellipsoid.get_xyz[0]
    B_ = B_[0]
    trA_1 = torch.trace(torch.linalg.inv(Aw))
    trA = torch.trace(Aw)
    detA = torch.linalg.det(Aw)

    B = s[..., :, None, None] * B_[..., None, :, :]
    mu = -torch.sqrt(torch.linalg.det(B) / detA)

    invB = torch.linalg.inv(B)
    trInvB = torch.vmap(torch.trace)(
        invB.view(-1, invB.shape[-2], invB.shape[-1])
    ).view(*invB.shape[:-2])
    trB = torch.vmap(torch.trace)(B.view(-1, B.shape[-2], B.shape[-1])).view(
        *B.shape[:-2]
    )

    V = torch.stack((trA_1 - mu * trInvB, 1 - mu, trB - mu * trA), dim=-1)[..., None]

    lambdaA, Pw_ell = torch.linalg.eig(Aw)
    lambdaA = torch.real(lambdaA)
    Pw_ell = torch.real(Pw_ell)
    M = torch.stack((torch.ones_like(lambdaA), lambdaA, torch.square(lambdaA)), dim=-2)
    D2 = torch.abs(torch.linalg.inv(M)[..., None, :, :] @ V)[..., 0].mT

    D = torch.empty(
        (*s.shape[:-1], 3, s.shape[-1] * 8), dtype=Aw.dtype, device=Aw.device
    )

    sqrt_D2 = torch.sqrt(D2)
    idx = torch.arange(0, s.shape[-1], dtype=torch.long, device=sqrt_D2.device)
    D[..., idx] = Pw_ell[..., :, :] @ (
        torch.tensor([1, 1, 1], dtype=sqrt_D2.dtype, device=sqrt_D2.device)[..., None]
        * sqrt_D2
    )
    D[..., (s.shape[-1] * 2 - 2) - (idx - 1)] = Pw_ell[..., :, :] @ (
        torch.tensor([1, 1, -1], dtype=sqrt_D2.dtype, device=sqrt_D2.device)[..., None]
        * sqrt_D2
    )
    D[..., (s.shape[-1] * 2) + idx] = Pw_ell[..., :, :] @ (
        torch.tensor([-1, 1, -1], dtype=sqrt_D2.dtype, device=sqrt_D2.device)[..., None]
        * sqrt_D2
    )
    D[..., (s.shape[-1] * 4 - 2) - (idx - 1)] = Pw_ell[..., :, :] @ (
        torch.tensor([-1, 1, 1], dtype=sqrt_D2.dtype, device=sqrt_D2.device)[..., None]
        * sqrt_D2
    )
    D[..., (s.shape[-1] * 4) + idx] = Pw_ell[..., :, :] @ (
        torch.tensor([1, -1, 1], dtype=sqrt_D2.dtype, device=sqrt_D2.device)[..., None]
        * sqrt_D2
    )
    D[..., (s.shape[-1] * 6 - 2) - (idx - 1)] = Pw_ell[..., :, :] @ (
        torch.tensor([1, -1, -1], dtype=sqrt_D2.dtype, device=sqrt_D2.device)[..., None]
        * sqrt_D2
    )
    D[..., s.shape[-1] * 6 + idx] = Pw_ell[..., :, :] @ (
        torch.tensor([-1, -1, -1], dtype=sqrt_D2.dtype, device=sqrt_D2.device)[
            ..., None
        ]
        * sqrt_D2
    )
    D[..., (s.shape[-1] * 8 - 2) - (idx - 1)] = Pw_ell[..., :, :] @ (
        torch.tensor([-1, -1, 1], dtype=sqrt_D2.dtype, device=sqrt_D2.device)[..., None]
        * sqrt_D2
    )

    Ew = (Cw[..., :, None] + D).mT
    s_Ew = (
        torch.concatenate((s, torch.flip(s, dims=(-1,))), dim=-1)[..., None]
        .expand(-1, 4)
        .reshape(*Ew.shape[:-2], -1, 1)
    )
    return Ew, s_Ew


def Aell_method():
    lambdaA = torch.linalg.eigvals(A)
    M = torch.tensor([[1, 1, 1], lambdaA, lambdaA**2])
    invM = torch.linalg.inv(M)
    trB_ = torch.trace(B_)
    trB__1 = torch.trace(torch.linalg.inv(B_))
    detB_ = torch.linalg.det(B_)
    trA = torch.trace(A)
    trA_1 = torch.trace(torch.linalg.inv(A))
    detA = torch.linalg.det(A)

    if detB_ < 0:  # sigma < 0
        mu0 = torch.sqrt(-detB_ / detA)
        D2 = torch.zeros((3, 4))
        rtsD2 = torch.zeros((3, 3))
        def_ = torch.zeros((6, 2))
        for i in range(3):
            D2[i, :] = [
                mu0 * (invM[i, 1] + invM[i, 2] * trA),
                -invM[i, 2] * trB_,
                -invM[i, 0] * mu0 * trB__1,
                invM[i, 0] * trA_1 + invM[i, 1],
            ]
            tmp = torch.roots(D2[i, :])
            tmp[tmp < 0] = torch.nan
            rtsD2[i, :] = -(tmp**2)
            if (
                D2[i, 0] < 0
            ):  # D2i(sigma) increases/negative first (decreases when sigma --> -Inf)
                def_[2 * i : 2 * i + 2, :] = [
                    [min(rtsD2[i, 0], 0), min(rtsD2[i, 1], 0)],
                    [min(rtsD2[i, 2], 0), 0],
                ]
            else:  # D2i(sigma) decreases/positive first (increases when sigma --> -Inf)
                def_[2 * i : 2 * i + 2, :] = [
                    [-torch.inf, min(rtsD2[i, 0], 0)],
                    [min(rtsD2[i, 1], 0), min(rtsD2[i, 2], 0)],
                ]

        if sum([row[0] == float("-inf") for row in def_]) == 3:
            raise ValueError("Domain error for sigma.")
        else:
            MINI = []
            for j in range(0, 6):
                tmp = def_[j][0]
                if j in [1, 2]:
                    c1 = (def_[2][0] < tmp < def_[2][1]) or (
                        def_[3][0] < tmp < def_[3][1]
                    )
                    c2 = (def_[4][0] < tmp < def_[4][1]) or (
                        def_[5][0] < tmp < def_[5][1]
                    )
                    if c1 and c2:
                        MINI.append(tmp)
                elif j in [3, 4]:
                    c1 = (def_[0][0] < tmp < def_[0][1]) or (
                        def_[1][0] < tmp < def_[1][1]
                    )
                    c2 = (def_[4][0] < tmp < def_[4][1]) or (
                        def_[5][0] < tmp < def_[5][1]
                    )
                    if c1 and c2:
                        MINI.append(tmp)
                elif j in [5, 6]:
                    c1 = (def_[0][0] < tmp < def_[0][1]) or (
                        def_[1][0] < tmp < def_[1][1]
                    )
                    c2 = (def_[2][0] < tmp < def_[2][1]) or (
                        def_[3][0] < tmp < def_[3][1]
                    )
                    if c1 and c2:
                        MINI.append(tmp)

            MAXI = []
            for j in range(0, 6):
                tmp = def_[j][1]
                if j in [1, 2]:
                    c1 = (def_[2][0] < tmp < def_[2][1]) or (
                        def_[3][0] < tmp < def_[3][1]
                    )
                    c2 = (def_[4][0] < tmp < def_[4][1]) or (
                        def_[5][0] < tmp < def_[5][1]
                    )
                    if c1 and c2:
                        MAXI.append(tmp)
                elif j in [3, 4]:
                    c1 = (def_[0][0] < tmp < def_[0][1]) or (
                        def_[1][0] < tmp < def_[1][1]
                    )
                    c2 = (def_[4][0] < tmp < def_[4][1]) or (
                        def_[5][0] < tmp < def_[5][1]
                    )
                    if c1 and c2:
                        MAXI.append(tmp)
                elif j in [5, 6]:
                    c1 = (def_[0][0] < tmp < def_[0][1]) or (
                        def_[1][0] < tmp < def_[1][1]
                    )
                    c2 = (def_[2][0] < tmp < def_[2][1]) or (
                        def_[3][0] < tmp < def_[3][1]
                    )
                    if c1 and c2:
                        MAXI.append(tmp)

            if (len(MINI) > 1) or (len(MAXI) > 1):
                raise ValueError("Domain error for sigma (2).")
            else:
                if (len(MINI) == 0) or (len(MAXI) == 0):
                    Domain = []
                else:
                    Domain = (
                        torch.linspace(0.0, 1.0, 1000, dtype=A.dtype, device=A.device)
                        * (MAXI[0] - MINI[0])
                        + MINI[0]
                    )

    else:
        mu0 = torch.sqrt(torch.linalg.det(B_) / torch.linalg.det(A))
        D2 = torch.zeros((3, 4))
        rtsD2 = torch.zeros((3, 3))
        def_ = torch.zeros((6, 4))

        for i in range(3):
            D2[i, :] = [
                mu0 * (invM[i, 1] + invM[i, 2] * trA),
                invM[i, 2] * trB_,
                invM[i, 0] * mu0 * trB__1,
                invM[i, 0] * trA_1 + invM[i, 1],
            ]
            tmp = torch.roots(D2[i, :])
            tmp[tmp < 0] = torch.nan
            rtsD2[i, :] = tmp**2
            if D2[i, 0] < 0:
                def_[(i - 1) * 2 : (i - 1) * 2 + 2, :] = [
                    [
                        0,
                        max(rtsD2[i, 0], 0),
                        max(rtsD2[i, 1], 0),
                        max(rtsD2[i, 2], 0),
                    ],
                    [max(rtsD2[i, 3], 0), torch.inf, torch.inf, torch.inf],
                ]
            else:
                def_[(i - 1) * 2 : (i - 1) * 2 + 2, :] = [
                    [
                        max(rtsD2[i, 0], 0),
                        max(rtsD2[i, 1], 0),
                        max(rtsD2[i, 2], 0),
                        torch.inf,
                    ],
                    [max(rtsD2[i, 3], 0), torch.inf, torch.inf, torch.inf],
                ]

        if torch.sum(def_[:, 3] == torch.inf) == 3:
            raise ValueError("Domain error for sigma.")
        else:
            MINI = []
            for j in range(0, 6):
                tmp = def_[j][0]
                if j in [1, 2]:
                    c1 = (def_[2][0] < tmp < def_[2][1]) or (
                        def_[3][0] < tmp < def_[3][1]
                    )
                    c2 = (def_[4][0] < tmp < def_[4][1]) or (
                        def_[5][0] < tmp < def_[5][1]
                    )
                    if c1 and c2:
                        MINI.append(tmp)
                elif j in [3, 4]:
                    c1 = (def_[0][0] < tmp < def_[0][1]) or (
                        def_[1][0] < tmp < def_[1][1]
                    )
                    c2 = (def_[4][0] < tmp < def_[4][1]) or (
                        def_[5][0] < tmp < def_[5][1]
                    )
                    if c1 and c2:
                        MINI.append(tmp)
                elif j in [5, 6]:
                    c1 = (def_[0][0] < tmp < def_[0][1]) or (
                        def_[1][0] < tmp < def_[1][1]
                    )
                    c2 = (def_[2][0] < tmp < def_[2][1]) or (
                        def_[3][0] < tmp < def_[3][1]
                    )
                    if c1 and c2:
                        MINI.append(tmp)

            MAXI = []
            for j in range(0, 6):
                tmp = def_[j][1]
                if j in [1, 2]:
                    c1 = (def_[2][0] < tmp < def_[2][1]) or (
                        def_[3][0] < tmp < def_[3][1]
                    )
                    c2 = (def_[4][0] < tmp < def_[4][1]) or (
                        def_[5][0] < tmp < def_[5][1]
                    )
                    if c1 and c2:
                        MAXI.append(tmp)
                elif j in [3, 4]:
                    c1 = (def_[0][0] < tmp < def_[0][1]) or (
                        def_[1][0] < tmp < def_[1][1]
                    )
                    c2 = (def_[4][0] < tmp < def_[4][1]) or (
                        def_[5][0] < tmp < def_[5][1]
                    )
                    if c1 and c2:
                        MAXI.append(tmp)
                elif j in [5, 6]:
                    c1 = (def_[0][0] < tmp < def_[0][1]) or (
                        def_[1][0] < tmp < def_[1][1]
                    )
                    c2 = (def_[2][0] < tmp < def_[2][1]) or (
                        def_[3][0] < tmp < def_[3][1]
                    )
                    if c1 and c2:
                        MAXI.append(tmp)

            if (len(MINI) > 1) or (len(MAXI) > 1):
                raise ValueError("Domain error for sigma (2).")
            else:
                if (len(MINI) == 0) or (len(MAXI) == 0):
                    Domain = []
                else:
                    Domain = (
                        torch.linspace(0.0, 1.0, 1000, dtype=A.dtype, device=A.device)
                        * (MAXI[0] - MINI[0])
                        + MINI[0]
                    )


def get_sigma_domain(A, B_):
    # A: ellipsoid orientation & shape
    # B_: cone orientation & shape
    # Domain: interval where sigma (scalar unknown in the Cone Alignment Equation) is defined
    method = "B_cone"  # 'Aell' or 'B_cone'

    if method == "Aell":
        lambdaA = torch.linalg.eigvals(A)
        M = torch.tensor([[1, 1, 1], lambdaA, lambdaA**2])
        invM = torch.linalg.inv(M)
        trB_ = torch.trace(B_)
        trB__1 = torch.trace(torch.linalg.inv(B_))
        detB_ = torch.linalg.det(B_)
        trA = torch.trace(A)
        trA_1 = torch.trace(torch.linalg.inv(A))
        detA = torch.linalg.det(A)

        if detB_ < 0:  # sigma < 0
            mu0 = torch.sqrt(-detB_ / detA)
            D2 = torch.zeros((3, 4))
            rtsD2 = torch.zeros((3, 3))
            def_ = torch.zeros((6, 2))
            for i in range(3):
                D2[i, :] = [
                    mu0 * (invM[i, 1] + invM[i, 2] * trA),
                    -invM[i, 2] * trB_,
                    -invM[i, 0] * mu0 * trB__1,
                    invM[i, 0] * trA_1 + invM[i, 1],
                ]
                tmp = torch.roots(D2[i, :])
                tmp[tmp < 0] = torch.nan
                rtsD2[i, :] = -(tmp**2)
                if (
                    D2[i, 0] < 0
                ):  # D2i(sigma) increases/negative first (decreases when sigma --> -Inf)
                    def_[2 * i : 2 * i + 2, :] = [
                        [min(rtsD2[i, 0], 0), min(rtsD2[i, 1], 0)],
                        [min(rtsD2[i, 2], 0), 0],
                    ]
                else:  # D2i(sigma) decreases/positive first (increases when sigma --> -Inf)
                    def_[2 * i : 2 * i + 2, :] = [
                        [-torch.inf, min(rtsD2[i, 0], 0)],
                        [min(rtsD2[i, 1], 0), min(rtsD2[i, 2], 0)],
                    ]

            if sum([row[0] == float("-inf") for row in def_]) == 3:
                raise ValueError("Domain error for sigma.")
            else:
                MINI = []
                for j in range(0, 6):
                    tmp = def_[j][0]
                    if j in [1, 2]:
                        c1 = (def_[2][0] < tmp < def_[2][1]) or (
                            def_[3][0] < tmp < def_[3][1]
                        )
                        c2 = (def_[4][0] < tmp < def_[4][1]) or (
                            def_[5][0] < tmp < def_[5][1]
                        )
                        if c1 and c2:
                            MINI.append(tmp)
                    elif j in [3, 4]:
                        c1 = (def_[0][0] < tmp < def_[0][1]) or (
                            def_[1][0] < tmp < def_[1][1]
                        )
                        c2 = (def_[4][0] < tmp < def_[4][1]) or (
                            def_[5][0] < tmp < def_[5][1]
                        )
                        if c1 and c2:
                            MINI.append(tmp)
                    elif j in [5, 6]:
                        c1 = (def_[0][0] < tmp < def_[0][1]) or (
                            def_[1][0] < tmp < def_[1][1]
                        )
                        c2 = (def_[2][0] < tmp < def_[2][1]) or (
                            def_[3][0] < tmp < def_[3][1]
                        )
                        if c1 and c2:
                            MINI.append(tmp)

                MAXI = []
                for j in range(0, 6):
                    tmp = def_[j][1]
                    if j in [1, 2]:
                        c1 = (def_[2][0] < tmp < def_[2][1]) or (
                            def_[3][0] < tmp < def_[3][1]
                        )
                        c2 = (def_[4][0] < tmp < def_[4][1]) or (
                            def_[5][0] < tmp < def_[5][1]
                        )
                        if c1 and c2:
                            MAXI.append(tmp)
                    elif j in [3, 4]:
                        c1 = (def_[0][0] < tmp < def_[0][1]) or (
                            def_[1][0] < tmp < def_[1][1]
                        )
                        c2 = (def_[4][0] < tmp < def_[4][1]) or (
                            def_[5][0] < tmp < def_[5][1]
                        )
                        if c1 and c2:
                            MAXI.append(tmp)
                    elif j in [5, 6]:
                        c1 = (def_[0][0] < tmp < def_[0][1]) or (
                            def_[1][0] < tmp < def_[1][1]
                        )
                        c2 = (def_[2][0] < tmp < def_[2][1]) or (
                            def_[3][0] < tmp < def_[3][1]
                        )
                        if c1 and c2:
                            MAXI.append(tmp)

                if (len(MINI) > 1) or (len(MAXI) > 1):
                    raise ValueError("Domain error for sigma (2).")
                else:
                    if (len(MINI) == 0) or (len(MAXI) == 0):
                        Domain = []
                    else:
                        Domain = (
                            torch.linspace(
                                0.0, 1.0, 1000, dtype=A.dtype, device=A.device
                            )
                            * (MAXI[0] - MINI[0])
                            + MINI[0]
                        )
        else:
            mu0 = torch.sqrt(torch.linalg.det(B_) / torch.linalg.det(A))
            D2 = torch.zeros((3, 4))
            rtsD2 = torch.zeros((3, 3))
            def_ = torch.zeros((6, 4))

            for i in range(3):
                D2[i, :] = [
                    mu0 * (invM[i, 1] + invM[i, 2] * trA),
                    invM[i, 2] * trB_,
                    invM[i, 0] * mu0 * trB__1,
                    invM[i, 0] * trA_1 + invM[i, 1],
                ]
                tmp = torch.roots(D2[i, :])
                tmp[tmp < 0] = torch.nan
                rtsD2[i, :] = tmp**2
                if D2[i, 0] < 0:
                    def_[(i - 1) * 2 : (i - 1) * 2 + 2, :] = [
                        [
                            0,
                            max(rtsD2[i, 0], 0),
                            max(rtsD2[i, 1], 0),
                            max(rtsD2[i, 2], 0),
                        ],
                        [max(rtsD2[i, 3], 0), torch.inf, torch.inf, torch.inf],
                    ]
                else:
                    def_[(i - 1) * 2 : (i - 1) * 2 + 2, :] = [
                        [
                            max(rtsD2[i, 0], 0),
                            max(rtsD2[i, 1], 0),
                            max(rtsD2[i, 2], 0),
                            torch.inf,
                        ],
                        [max(rtsD2[i, 3], 0), torch.inf, torch.inf, torch.inf],
                    ]

            if torch.sum(def_[:, 3] == torch.inf) == 3:
                raise ValueError("Domain error for sigma.")
            else:
                MINI = []
                for j in range(0, 6):
                    tmp = def_[j][0]
                    if j in [1, 2]:
                        c1 = (def_[2][0] < tmp < def_[2][1]) or (
                            def_[3][0] < tmp < def_[3][1]
                        )
                        c2 = (def_[4][0] < tmp < def_[4][1]) or (
                            def_[5][0] < tmp < def_[5][1]
                        )
                        if c1 and c2:
                            MINI.append(tmp)
                    elif j in [3, 4]:
                        c1 = (def_[0][0] < tmp < def_[0][1]) or (
                            def_[1][0] < tmp < def_[1][1]
                        )
                        c2 = (def_[4][0] < tmp < def_[4][1]) or (
                            def_[5][0] < tmp < def_[5][1]
                        )
                        if c1 and c2:
                            MINI.append(tmp)
                    elif j in [5, 6]:
                        c1 = (def_[0][0] < tmp < def_[0][1]) or (
                            def_[1][0] < tmp < def_[1][1]
                        )
                        c2 = (def_[2][0] < tmp < def_[2][1]) or (
                            def_[3][0] < tmp < def_[3][1]
                        )
                        if c1 and c2:
                            MINI.append(tmp)

                MAXI = []
                for j in range(0, 6):
                    tmp = def_[j][1]
                    if j in [1, 2]:
                        c1 = (def_[2][0] < tmp < def_[2][1]) or (
                            def_[3][0] < tmp < def_[3][1]
                        )
                        c2 = (def_[4][0] < tmp < def_[4][1]) or (
                            def_[5][0] < tmp < def_[5][1]
                        )
                        if c1 and c2:
                            MAXI.append(tmp)
                    elif j in [3, 4]:
                        c1 = (def_[0][0] < tmp < def_[0][1]) or (
                            def_[1][0] < tmp < def_[1][1]
                        )
                        c2 = (def_[4][0] < tmp < def_[4][1]) or (
                            def_[5][0] < tmp < def_[5][1]
                        )
                        if c1 and c2:
                            MAXI.append(tmp)
                    elif j in [5, 6]:
                        c1 = (def_[0][0] < tmp < def_[0][1]) or (
                            def_[1][0] < tmp < def_[1][1]
                        )
                        c2 = (def_[2][0] < tmp < def_[2][1]) or (
                            def_[3][0] < tmp < def_[3][1]
                        )
                        if c1 and c2:
                            MAXI.append(tmp)

                if (len(MINI) > 1) or (len(MAXI) > 1):
                    raise ValueError("Domain error for sigma (2).")
                else:
                    if (len(MINI) == 0) or (len(MAXI) == 0):
                        Domain = []
                    else:
                        Domain = (
                            torch.linspace(
                                0.0, 1.0, 1000, dtype=A.dtype, device=A.device
                            )
                            * (MAXI[0] - MINI[0])
                            + MINI[0]
                        )

    elif method == "B_cone":
        lambdaB_, _ = torch.linalg.eig(B_)
        lambdaB_ = torch.real(lambdaB_)
        lambdaB_ = torch.flip(lambdaB_, dims=(-1,))
        M = torch.tensor([[1, 1, 1], lambdaB_, lambdaB_**2])
        invM = torch.linalg.inv(M)
        trB_ = torch.trace(B_)
        trB__1 = torch.trace(torch.linalg.inv(B_))
        detB_ = torch.linalg.det(B_)
        trA = torch.trace(A)
        trA_1 = torch.trace(torch.linalg.inv(A))
        detA = torch.linalg.det(A)

        if detB_ < 0:
            mu0 = torch.sqrt(-detB_ / detA)
            D2 = torch.zeros((3, 4))
            rtsD2 = torch.zeros((3, 3))
            def_ = torch.zeros((6, 2))

            for i in range(3):
                D2[i, :] = torch.stack(
                    (
                        -mu0 * (invM[i, 0] * trB__1 + invM[i, 1]),
                        invM[i, 0] * trA_1,
                        mu0 * invM[i, 2] * trA,
                        -(invM[i, 2] * trB_ + invM[i, 1]),
                    ),
                    dim=-1,
                )
                tmp = torch.real(
                    torch.from_numpy(np.roots(D2[i, :].detach().cpu().numpy()))
                )
                tmp[tmp < 0] = torch.nan
                rtsD2[i, :] = -(tmp**2)

                if D2[i, 0] < 0:
                    def_[i * 2 : i * 2 + 2, :] = torch.stack(
                        (
                            torch.stack(
                                (
                                    torch.clamp_max(
                                        torch.nan_to_num(rtsD2[i, 0], nan=0.0), 0
                                    ),
                                    torch.clamp_max(
                                        torch.nan_to_num(rtsD2[i, 1], nan=0.0), 0
                                    ),
                                )
                            ),
                            torch.stack(
                                (
                                    torch.clamp_max(
                                        torch.nan_to_num(rtsD2[i, 2], nan=0.0), 0
                                    ),
                                    torch.tensor(
                                        0, dtype=rtsD2.dtype, device=rtsD2.device
                                    ),
                                ),
                            ),
                        ),
                        dim=-2,
                    )
                else:
                    def_[i * 2 : i * 2 + 2, :] = torch.stack(
                        (
                            torch.stack(
                                (
                                    torch.tensor(
                                        -torch.inf,
                                        dtype=rtsD2.dtype,
                                        device=rtsD2.device,
                                    ),
                                    torch.clamp_max(
                                        torch.nan_to_num(rtsD2[i, 0], nan=0.0), 0
                                    ),
                                )
                            ),
                            torch.stack(
                                (
                                    torch.clamp_max(
                                        torch.nan_to_num(rtsD2[i, 1], nan=0.0), 0
                                    ),
                                    torch.clamp_max(
                                        torch.nan_to_num(rtsD2[i, 2], nan=0.0), 0
                                    ),
                                )
                            ),
                        ),
                        dim=-2,
                    )
            if sum([row[0] == float("-inf") for row in def_]) == 3:
                raise ValueError("Domain error for sigma.")
            else:
                MINI = []
                for j in range(0, 6):
                    tmp = def_[j, 0]
                    if j in [0, 1]:
                        c1 = (def_[2, 0] < tmp < def_[2, 1]) or (
                            def_[3, 0] < tmp < def_[3, 1]
                        )
                        c2 = (def_[4, 0] < tmp < def_[4, 1]) or (
                            def_[5, 0] < tmp < def_[5, 1]
                        )
                        if c1 and c2:
                            MINI.append(tmp)
                    elif j in [2, 3]:
                        c1 = (def_[0, 0] < tmp < def_[0, 1]) or (
                            def_[1, 0] < tmp < def_[1, 1]
                        )
                        c2 = (def_[4, 0] < tmp < def_[4, 1]) or (
                            def_[5, 0] < tmp < def_[5, 1]
                        )
                        if c1 and c2:
                            MINI.append(tmp)
                    elif j in [4, 5]:
                        c1 = (def_[0, 0] < tmp < def_[0, 1]) or (
                            def_[1, 0] < tmp < def_[1, 1]
                        )
                        c2 = (def_[2, 0] < tmp < def_[2, 1]) or (
                            def_[3, 0] < tmp < def_[3, 1]
                        )
                        if c1 and c2:
                            MINI.append(tmp)

                MAXI = []
                for j in range(0, 6):
                    tmp = def_[j, 1]
                    if j in [0, 1]:
                        c1 = (def_[2, 0] < tmp < def_[2, 1]) or (
                            def_[3, 0] < tmp < def_[3, 1]
                        )
                        c2 = (def_[4, 0] < tmp < def_[4, 1]) or (
                            def_[5, 0] < tmp < def_[5, 1]
                        )
                        if c1 and c2:
                            MAXI.append(tmp)
                    elif j in [2, 3]:
                        c1 = (def_[0, 0] < tmp < def_[0, 1]) or (
                            def_[1, 0] < tmp < def_[1, 1]
                        )
                        c2 = (def_[4, 0] < tmp < def_[4, 1]) or (
                            def_[5, 0] < tmp < def_[5, 1]
                        )
                        if c1 and c2:
                            MAXI.append(tmp)
                    elif j in [4, 5]:
                        c1 = (def_[0, 0] < tmp < def_[0, 1]) or (
                            def_[1, 0] < tmp < def_[1, 1]
                        )
                        c2 = (def_[2, 0] < tmp < def_[2, 1]) or (
                            def_[3, 0] < tmp < def_[3, 1]
                        )
                        if c1 and c2:
                            MAXI.append(tmp)

                if (len(MINI) > 1) or (len(MAXI) > 1):
                    raise ValueError("Domain error for sigma (2).")
                else:
                    if (len(MINI) == 0) or (len(MAXI) == 0):
                        Domain = []
                    else:
                        Domain = (
                            torch.linspace(
                                0.0, 1.0, 1000, dtype=A.dtype, device=A.device
                            )
                            * (MAXI[0] - MINI[0])
                            + MINI[0]
                        )
        else:
            mu0 = torch.sqrt(torch.linalg.det(B_) / torch.linalg.det(A))
            D2 = torch.zeros((3, 4))
            rtsD2 = torch.zeros((3, 3))
            def_ = torch.zeros((6, 4))

            for i in range(3):
                D2[i, :] = torch.stack(
                    (
                        mu0 * (invM[i, 1] + invM[i, 2] * trA),
                        invM[i, 2] * trB_,
                        invM[i, 0] * mu0 * trB__1,
                        invM[i, 0] * trA_1 + invM[i, 1],
                    ),
                    dim=-1,
                )
                tmp = torch.real(
                    torch.from_numpy(np.roots(D2[i, :].detach().cpu().numpy()))
                )
                tmp[tmp < 0] = torch.nan
                rtsD2[i, :] = tmp**2
                if D2[i, 0] < 0:
                    def_[i * 2 : i * 2 + 2, :] = torch.stack(
                        (
                            torch.stack(
                                (
                                    torch.tensor(
                                        0, dtype=rtsD2.dtype, device=rtsD2.device
                                    ),
                                    torch.clamp_min(
                                        torch.nan_to_num(rtsD2[i, 0], nan=0.0), 0
                                    ),
                                    torch.clamp_min(
                                        torch.nan_to_num(rtsD2[i, 1], nan=0.0), 0
                                    ),
                                    torch.clamp_min(
                                        torch.nan_to_num(rtsD2[i, 2], nan=0.0), 0
                                    ),
                                )
                            ),
                            torch.stack(
                                (
                                    torch.clamp_min(
                                        torch.nan_to_num(rtsD2[i, 2], nan=0.0), 0
                                    ),
                                    torch.tensor(
                                        torch.inf,
                                        dtype=rtsD2.dtype,
                                        device=rtsD2.device,
                                    ),
                                    torch.tensor(
                                        torch.inf,
                                        dtype=rtsD2.dtype,
                                        device=rtsD2.device,
                                    ),
                                    torch.tensor(
                                        torch.inf,
                                        dtype=rtsD2.dtype,
                                        device=rtsD2.device,
                                    ),
                                ),
                            ),
                        ),
                        dim=-2,
                    )
                else:
                    def_[(i - 1) * 2 : (i - 1) * 2 + 2, :] = [
                        [
                            max(rtsD2[i, 0], 0),
                            max(rtsD2[i, 1], 0),
                            max(rtsD2[i, 2], 0),
                            torch.inf,
                        ],
                        [max(rtsD2[i, 3], 0), torch.inf, torch.inf, torch.inf],
                    ]
                    def_[i * 2 : i * 2 + 2, :] = torch.stack(
                        (
                            torch.stack(
                                (
                                    torch.clamp_max(
                                        torch.nan_to_num(rtsD2[i, 0], nan=0.0), 0
                                    ),
                                    torch.tensor(
                                        torch.inf,
                                        dtype=rtsD2.dtype,
                                        device=rtsD2.device,
                                    ),
                                )
                            ),
                            torch.stack(
                                (
                                    torch.clamp_max(
                                        torch.nan_to_num(rtsD2[i, 1], nan=0.0), 0
                                    ),
                                    torch.clamp_max(
                                        torch.nan_to_num(rtsD2[i, 2], nan=0.0), 0
                                    ),
                                )
                            ),
                        ),
                        dim=-2,
                    )

            if torch.sum(def_[:, 3] == torch.inf) == 3:
                raise ValueError("Domain error for sigma.")
            else:
                MINI = []
                for j in range(0, 6):
                    tmp = def_[j][0]
                    if j in [1, 2]:
                        c1 = (def_[2][0] < tmp < def_[2][1]) or (
                            def_[3][0] < tmp < def_[3][1]
                        )
                        c2 = (def_[4][0] < tmp < def_[4][1]) or (
                            def_[5][0] < tmp < def_[5][1]
                        )
                        if c1 and c2:
                            MINI.append(tmp)
                    elif j in [3, 4]:
                        c1 = (def_[0][0] < tmp < def_[0][1]) or (
                            def_[1][0] < tmp < def_[1][1]
                        )
                        c2 = (def_[4][0] < tmp < def_[4][1]) or (
                            def_[5][0] < tmp < def_[5][1]
                        )
                        if c1 and c2:
                            MINI.append(tmp)
                    elif j in [5, 6]:
                        c1 = (def_[0][0] < tmp < def_[0][1]) or (
                            def_[1][0] < tmp < def_[1][1]
                        )
                        c2 = (def_[2][0] < tmp < def_[2][1]) or (
                            def_[3][0] < tmp < def_[3][1]
                        )
                        if c1 and c2:
                            MINI.append(tmp)

                MAXI = []
                for j in range(0, 6):
                    tmp = def_[j][1]
                    if j in [1, 2]:
                        c1 = (def_[2][0] < tmp < def_[2][1]) or (
                            def_[3][0] < tmp < def_[3][1]
                        )
                        c2 = (def_[4][0] < tmp < def_[4][1]) or (
                            def_[5][0] < tmp < def_[5][1]
                        )
                        if c1 and c2:
                            MAXI.append(tmp)
                    elif j in [3, 4]:
                        c1 = (def_[0][0] < tmp < def_[0][1]) or (
                            def_[1][0] < tmp < def_[1][1]
                        )
                        c2 = (def_[4][0] < tmp < def_[4][1]) or (
                            def_[5][0] < tmp < def_[5][1]
                        )
                        if c1 and c2:
                            MAXI.append(tmp)
                    elif j in [5, 6]:
                        c1 = (def_[0][0] < tmp < def_[0][1]) or (
                            def_[1][0] < tmp < def_[1][1]
                        )
                        c2 = (def_[2][0] < tmp < def_[2][1]) or (
                            def_[3][0] < tmp < def_[3][1]
                        )
                        if c1 and c2:
                            MAXI.append(tmp)

                if (len(MINI) > 1) or (len(MAXI) > 1):
                    raise ValueError("Domain error for sigma (2).")
                else:
                    if (len(MINI) == 0) or (len(MAXI) == 0):
                        Domain = []
                    else:
                        Domain = (
                            torch.linspace(
                                0.0, 1.0, 1000, dtype=A.dtype, device=A.device
                            )
                            * (MAXI[0] - MINI[0])
                            + MINI[0]
                        )

    return Domain


def get_sigma_domain_parallel(A, B_):
    # A: ellipsoid orientation & shape
    # B_: cone orientation & shape
    # Domain: interval where sigma (scalar unknown in the Cone Alignment Equation) is defined
    # method = "B_cone"  # 'Aell' or 'B_cone'

    _, lambdaB_ = torch.linalg.eig(B_)
    lambdaB_ = torch.real(lambdaB_)
    M = torch.stack(
        (
            torch.ones(*lambdaB_.shape[:-1], 3, dtype=B_.dtype, device=B_.device),
            lambdaB_,
            lambdaB_**2,
        ),
        dim=-2,
    )
    invM = torch.linalg.inv(M)
    trB_ = torch.vmap(torch.trace)(B_)
    trB__1 = torch.vmap(torch.trace)(torch.linalg.inv(B_))
    detB_ = torch.linalg.det(B_)
    trA = torch.vmap(torch.trace)(A)
    trA_1 = torch.vmap(torch.trace)(torch.linalg.inv(A))
    detA = torch.linalg.det(A)

    negative_det_mask = detB_ < 0
    # here handle all the ones with positive determinant
    # D2 = torch.zeros(
    #     (torch.count_nonzero(negative_det_mask), 3, 4),
    #     dtype=B_.dtype,
    #     device=B_.device,
    # )
    rtsD2 = torch.zeros(
        (torch.count_nonzero(negative_det_mask), 3, 3),
        dtype=B_.dtype,
        device=B_.device,
    )
    def_ = torch.zeros(
        (torch.count_nonzero(negative_det_mask), 6, 2),
        dtype=B_.dtype,
        device=B_.device,
    )

    mu0 = torch.sqrt(-detB_[negative_det_mask] / detA[negative_det_mask])

    D2 = torch.stack(
        (
            -mu0[:, None]
            * (
                invM[negative_det_mask, :, 0] * trB__1[negative_det_mask, None]
                + invM[negative_det_mask, :, 1]
            ),
            invM[negative_det_mask, :, 0] * trA_1[negative_det_mask, None],
            mu0[:, None] * invM[negative_det_mask, :, 2] * trA[negative_det_mask, None],
            -(
                invM[negative_det_mask, :, 2] * trB_[negative_det_mask, None]
                + invM[negative_det_mask, :, 1]
            ),
        ),
        dim=-1,
    )

    tmp = torch.real(roots(D2))
    tmp[tmp < 0] = torch.nan
    rtsD2 = -torch.square(tmp)
    breakpoint()

    for i in range(3):
        if D2[i, 0] < 0:
            def_[(i - 1) * 2 : (i - 1) * 2 + 2, :] = [
                [min(rtsD2[i, 0], 0), min(rtsD2[i, 1], 0)],
                [min(rtsD2[i, 2], 0), 0],
            ]
        else:
            def_[(i - 1) * 2 : (i - 1) * 2 + 2, :] = [
                [-torch.inf, min(rtsD2[i, 0], 0)],
                [min(rtsD2[i, 1], 0), min(rtsD2[i, 2], 0)],
            ]
    if sum([row[0] == float("-inf") for row in def_]) == 3:
        raise ValueError("Domain error for sigma.")
    else:
        MINI = []
        for j in range(0, 6):
            tmp = def_[j][0]
            if j in [1, 2]:
                c1 = (def_[2][0] < tmp < def_[2][1]) or (def_[3][0] < tmp < def_[3][1])
                c2 = (def_[4][0] < tmp < def_[4][1]) or (def_[5][0] < tmp < def_[5][1])
                if c1 and c2:
                    MINI.append(tmp)
            elif j in [3, 4]:
                c1 = (def_[0][0] < tmp < def_[0][1]) or (def_[1][0] < tmp < def_[1][1])
                c2 = (def_[4][0] < tmp < def_[4][1]) or (def_[5][0] < tmp < def_[5][1])
                if c1 and c2:
                    MINI.append(tmp)
            elif j in [5, 6]:
                c1 = (def_[0][0] < tmp < def_[0][1]) or (def_[1][0] < tmp < def_[1][1])
                c2 = (def_[2][0] < tmp < def_[2][1]) or (def_[3][0] < tmp < def_[3][1])
                if c1 and c2:
                    MINI.append(tmp)

        MAXI = []
        for j in range(0, 6):
            tmp = def_[j][1]
            if j in [1, 2]:
                c1 = (def_[2][0] < tmp < def_[2][1]) or (def_[3][0] < tmp < def_[3][1])
                c2 = (def_[4][0] < tmp < def_[4][1]) or (def_[5][0] < tmp < def_[5][1])
                if c1 and c2:
                    MAXI.append(tmp)
            elif j in [3, 4]:
                c1 = (def_[0][0] < tmp < def_[0][1]) or (def_[1][0] < tmp < def_[1][1])
                c2 = (def_[4][0] < tmp < def_[4][1]) or (def_[5][0] < tmp < def_[5][1])
                if c1 and c2:
                    MAXI.append(tmp)
            elif j in [5, 6]:
                c1 = (def_[0][0] < tmp < def_[0][1]) or (def_[1][0] < tmp < def_[1][1])
                c2 = (def_[2][0] < tmp < def_[2][1]) or (def_[3][0] < tmp < def_[3][1])
                if c1 and c2:
                    MAXI.append(tmp)

        if (len(MINI) > 1) or (len(MAXI) > 1):
            raise ValueError("Domain error for sigma (2).")
        else:
            if (len(MINI) == 0) or (len(MAXI) == 0):
                Domain = []
            else:
                Domain = torch.linspace(MINI, MAXI, 1000)

    positive_det_mask = ~negative_det_mask
    # now handle the ones with negative determinant
    mu0 = torch.sqrt(torch.linalg.det(B_) / torch.linalg.det(A))
    D2 = torch.zeros((3, 4), dtype=B_.dtype, device=B_.device)[None]
    rtsD2 = torch.zeros((3, 3), dtype=B_.dtype, device=B_.device)[None]
    def_ = torch.zeros((6, 4), dtype=B_.dtype, device=B_.device)[None]

    for i in range(3):
        D2[i, :] = [
            mu0 * (invM[i, 1] + invM[i, 2] * trA),
            invM[i, 2] * trB_,
            invM[i, 0] * mu0 * trB__1,
            invM[i, 0] * trA_1 + invM[i, 1],
        ]
        tmp = torch.roots(D2[i, :])
        tmp[tmp < 0] = torch.nan
        rtsD2[i, :] = tmp**2
        if D2[i, 0] < 0:
            def_[(i - 1) * 2 : (i - 1) * 2 + 2, :] = [
                [
                    0,
                    max(rtsD2[i, 0], 0),
                    max(rtsD2[i, 1], 0),
                    max(rtsD2[i, 2], 0),
                ],
                [max(rtsD2[i, 3], 0), torch.inf, torch.inf, torch.inf],
            ]
        else:
            def_[(i - 1) * 2 : (i - 1) * 2 + 2, :] = [
                [
                    max(rtsD2[i, 0], 0),
                    max(rtsD2[i, 1], 0),
                    max(rtsD2[i, 2], 0),
                    torch.inf,
                ],
                [max(rtsD2[i, 3], 0), torch.inf, torch.inf, torch.inf],
            ]

    if torch.sum(def_[:, 3] == torch.inf) == 3:
        raise ValueError("Domain error for sigma.")
    else:
        MINI = []
        for j in range(0, 6):
            tmp = def_[j][0]
            if j in [1, 2]:
                c1 = (def_[2][0] < tmp < def_[2][1]) or (def_[3][0] < tmp < def_[3][1])
                c2 = (def_[4][0] < tmp < def_[4][1]) or (def_[5][0] < tmp < def_[5][1])
                if c1 and c2:
                    MINI.append(tmp)
            elif j in [3, 4]:
                c1 = (def_[0][0] < tmp < def_[0][1]) or (def_[1][0] < tmp < def_[1][1])
                c2 = (def_[4][0] < tmp < def_[4][1]) or (def_[5][0] < tmp < def_[5][1])
                if c1 and c2:
                    MINI.append(tmp)
            elif j in [5, 6]:
                c1 = (def_[0][0] < tmp < def_[0][1]) or (def_[1][0] < tmp < def_[1][1])
                c2 = (def_[2][0] < tmp < def_[2][1]) or (def_[3][0] < tmp < def_[3][1])
                if c1 and c2:
                    MINI.append(tmp)

        MAXI = []
        for j in range(0, 6):
            tmp = def_[j][1]
            if j in [1, 2]:
                c1 = (def_[2][0] < tmp < def_[2][1]) or (def_[3][0] < tmp < def_[3][1])
                c2 = (def_[4][0] < tmp < def_[4][1]) or (def_[5][0] < tmp < def_[5][1])
                if c1 and c2:
                    MAXI.append(tmp)
            elif j in [3, 4]:
                c1 = (def_[0][0] < tmp < def_[0][1]) or (def_[1][0] < tmp < def_[1][1])
                c2 = (def_[4][0] < tmp < def_[4][1]) or (def_[5][0] < tmp < def_[5][1])
                if c1 and c2:
                    MAXI.append(tmp)
            elif j in [5, 6]:
                c1 = (def_[0][0] < tmp < def_[0][1]) or (def_[1][0] < tmp < def_[1][1])
                c2 = (def_[2][0] < tmp < def_[2][1]) or (def_[3][0] < tmp < def_[3][1])
                if c1 and c2:
                    MAXI.append(tmp)

        if (len(MINI) > 1) or (len(MAXI) > 1):
            raise ValueError("Domain error for sigma (2).")
        else:
            if (len(MINI) == 0) or (len(MAXI) == 0):
                Domain = []
            else:
                Domain = torch.linspace(MINI, MAXI, 1000)

    return Domain


def compute_backproj_cone(
    ellipsis_model: EllipsisModel, camera_intrinsics: torch.Tensor
):
    """Compute back-projection cone corresponding to Ellipse"""
    R = (
        ellipsis_model.get_rotation_mat.mT
    )  # rotation matrix from the ellipsoid are transpose as this is recreate from zero we need to transpose it
    scale = ellipsis_model.get_scaling
    a = scale[..., 0]
    b = scale[..., 1]
    means_2D = ellipsis_model.get_xy

    means_2D_hom = torch.cat(
        (
            means_2D,
            torch.ones(*means_2D.shape[:-1], 1, dtype=scale.dtype, device=scale.device),
        ),
        dim=-1,
    )
    # camera_f = (camera_intrinsics[..., 0, 0] + camera_intrinsics[..., 1, 1]) / 2.0
    Kc = torch.linalg.inv(camera_intrinsics) @ means_2D_hom[..., :, None]

    return compute_backproj_cone_internally(
        Kc,
        R,
        a,
        b,
    )


def compute_backproj_cone_internally(
    Kc: torch.Tensor,
    R: torch.Tensor,
    a: torch.Tensor,
    b: torch.Tensor,
):
    Ec = torch.tensor([0, 0, 0], dtype=Kc.dtype, device=Kc.device)[None, :, None]
    Uc = torch.concatenate(
        (
            R[..., :, 0],
            torch.zeros(*R.shape[:-2], 1, dtype=Kc.dtype, device=Kc.device),
        ),
        dim=-1,
    )[..., :, None]
    Vc = torch.concatenate(
        (
            R[..., :, 1],
            torch.zeros(*R.shape[:-2], 1, dtype=Kc.dtype, device=Kc.device),
        ),
        dim=-1,
    )[..., :, None]

    Nc = torch.tensor([[[0], [0], [1]]], dtype=Kc.dtype, device=Kc.device)
    M = (Uc @ Uc.mT) / a[..., None, None] + (Vc @ Vc.mT) / b[..., None, None]
    W = Nc / (Nc.mT @ (Kc - Ec))
    P = torch.eye(3, dtype=Kc.dtype, device=Kc.device)[None] - ((Kc - Ec) @ W.mT)
    B_ = (P.mT @ M @ P) - (W @ W.mT)
    return B_
