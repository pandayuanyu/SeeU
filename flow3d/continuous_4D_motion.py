import torch
import torch.nn as nn
import torch.nn.functional as F
from loguru import logger as guru

def t_norm_to_unit(t_norm: torch.Tensor) -> torch.Tensor:
    """
    [-1,1] -> (0,1), avoiding the endpoints
    """
    return (t_norm * 0.5 + 0.5).clamp(1e-6, 1 - 1e-6)

# ====================== B-spline basis functions ======================

class BSplineBasis(nn.Module):
    """
    Generate a [B, m] B-spline basis matrix.
    Implemented using the built-in Cox–de Boor recursion formula.
    """
    def __init__(self, num_ctrl: int, degree: int):
        super().__init__()
        assert num_ctrl >= degree + 1, "Number of control points must be >= degree+1"
        self.m = num_ctrl
        self.p = degree
        
        # Open-uniform knots
        knots = torch.zeros(self.m + self.p + 1)
        knots[self.p:self.m+1] = torch.linspace(0., 1., self.m - self.p + 1)
        knots[self.m+1:] = 1.
        self.register_buffer("knots", knots)

    def forward(self, t01: torch.Tensor) -> torch.Tensor:
        # t01: [B, 1] in (0,1)
        
        # fallback: Cox–de Boor
        u = t01.squeeze(-1)           # [B]
        B, m, p, knots = u.shape[0], self.m, self.p, self.knots

        def N(i, k, x):
            # N_{i,0}
            if k == 0:
                left = (knots[i] <= x)
                # Use a closed interval for the last knot interval [t_m, t_m+1]
                if i+1 == len(knots)-1: 
                    right = (x <= knots[i+1])
                else:
                    right = (x < knots[i+1])
                return (left & right).float()
            
            # recursion
            denom1 = knots[i+k] - knots[i]
            denom2 = knots[i+k+1] - knots[i+1]
            term1 = 0.0
            term2 = 0.0
            if denom1 > 1e-8: # improved numerical stability
                term1 = (x - knots[i]) / denom1 * N(i, k-1, x)
            if denom2 > 1e-8: # improved numerical stability
                term2 = (knots[i+k+1] - x) / denom2 * N(i+1, k-1, x)
            return term1 + term2

        cols = [N(i, p, u) for i in range(m)]
        return torch.stack(cols, dim=1)  # [B, m]

# ====================== Continuous model (learnable control points) ======================
class ContinuousMotionBases(nn.Module):
    """
    Output [K, B, 9], where each base contains (rot6d[6] + trans[3]).
    Control points have shape [K, 9, m], and the time basis matrix is [B, m].
    """
    def __init__(self, num_bases:int, num_frames:int, num_control_points:int, degree:int=3):
        super().__init__()
        self.K = num_bases
        self.T = num_frames
        self.m = num_control_points
        self.p = degree
        self.control_points = nn.Parameter(torch.randn(self.K, 9, self.m) * 0.05)
        self.bspline = BSplineBasis(self.m, self.p)

    def forward(self, t_norm: torch.Tensor) -> torch.Tensor:
        """
        Standard forward pass: stop gradients through the B-spline basis,
        and optimize only the control points.
        Input t_norm: [B,1] in [-1,1]; output [K,B,9]
        """
        t01 = t_norm_to_unit(t_norm)
        # Only treat the B-spline as a fixed basis function
        self.bspline.eval()
        with torch.no_grad():
            Bmat = self.bspline(t01)                         # [B, m]
        Bmat = Bmat.to(self.control_points.dtype).to(self.control_points.device)
        out = self.control_points @ Bmat.T             # (K,9,m) @ (m,B) -> (K,9,B)
        return out.transpose(1, 2)                     # [K,B,9]

    def forward_extrap(self, t_norm: torch.Tensor) -> torch.Tensor:
        """
        Same output shape as forward: [K,B,9], but applies linear extrapolation
        for t_norm outside [-1,1], with slopes estimated from endpoint finite differences
        using a 1-frame step size.
        """
        device = self.control_points.device
        t_norm = t_norm.to(device)

        # First get the normal output clamped to [-1,1] (endpoint values)
        t_clamp = t_norm.clamp(-1.0, 1.0)
        y0 = self.forward(t_clamp)                         # [K,B,9]

        # First-order endpoint derivatives (finite differences), normalized length of 1 frame
        T1  = max(self.T - 1, 1)
        eps = 2.0 / T1 - 1e-12

        # Left endpoint slope
        tL  = torch.tensor([[-1.0]], device=device)
        tLh = torch.tensor([[-1.0 + eps]], device=device)
        yL  = self.forward(tL)                             # [K,1,9]
        yLh = self.forward(tLh)                            # [K,1,9]
        slopeL = (yLh - yL) / eps                           # [K,1,9]

        # Right endpoint slope (using backward difference)
        tR  = torch.tensor([[ 1.0]], device=device)
        tRh = torch.tensor([[ 1.0 - eps]], device=device)
        yR  = self.forward(tR)                             # [K,1,9]
        yRh = self.forward(tRh)                            # [K,1,9]
        slopeR = (yR - yRh) / eps                           # [K,1,9]

        # Assemble according to the interval where t lies
        tn = t_norm.squeeze(-1)                           # [B]
        out = y0.clone()

        idxL = torch.nonzero(tn < -1.0, as_tuple=False).squeeze(-1)
        if idxL.numel() > 0:
            d = (tn[idxL] + 1.0).view(1, -1, 1)         # [1,BL,1] = t - (-1)
            out[:, idxL, :] = yL.expand(-1, idxL.numel(), -1) + slopeL.expand(-1, idxL.numel(), -1) * d

        idxR = torch.nonzero(tn >  1.0, as_tuple=False).squeeze(-1)
        if idxR.numel() > 0:
            d = (tn[idxR] - 1.0).view(1, -1, 1)         # [1,BR,1] = t - ( 1)
            out[:, idxR, :] = yR.expand(-1, idxR.numel(), -1) + slopeR.expand(-1, idxR.numel(), -1) * d

        return out


class ContinuousCameraPose(nn.Module):
    """
    Output [B, 9], with control points of shape [1, 9, m]
    """
    def __init__(self, num_frames:int, num_control_points:int, degree:int=3):
        super().__init__()
        self.T = num_frames
        self.m = num_control_points
        self.p = degree
        self.control_points = nn.Parameter(torch.randn(1, 9, self.m) * 0.05)
        self.bspline = BSplineBasis(self.m, self.p)

    def forward(self, t_norm: torch.Tensor) -> torch.Tensor:
        """
        Standard forward pass: stop gradients through the B-spline basis,
        and optimize only the control points.
        Input t_norm: [B,1] in [-1,1]; output [B,9]
        """
        t01 = t_norm_to_unit(t_norm)
        self.bspline.eval()
        with torch.no_grad():
            Bmat = self.bspline(t01)                         # [B, m]
        Bmat = Bmat.to(self.control_points.dtype).to(self.control_points.device)
        out = self.control_points @ Bmat.T             # (1,9,m) @ (m,B) -> (1,9,B)
        return out.squeeze(0).transpose(0, 1)              # [B,9]

    def forward_extrap(self, t_norm: torch.Tensor) -> torch.Tensor:
        """
        Same output shape as forward: [B,9], but applies linear extrapolation
        for t_norm outside [-1,1].
        """
        device = self.control_points.device
        t_norm = t_norm.to(device)

        t_clamp = t_norm.clamp(-1.0, 1.0)
        y0 = self.forward(t_clamp)                         # [B,9]

        T1  = max(self.T - 1, 1)
        eps = 2.0 / T1 - 1e-12

        # Left endpoint
        tL  = torch.tensor([[-1.0]], device=device)
        tLh = torch.tensor([[-1.0 + eps]], device=device)
        yL  = self.forward(tL)                             # [1,9]
        yLh = self.forward(tLh)                            # [1,9]
        slopeL = (yLh - yL) / eps                           # [1,9]

        # Right endpoint (backward difference)
        tR  = torch.tensor([[ 1.0]], device=device)
        tRh = torch.tensor([[ 1.0 - eps]], device=device)
        yR  = self.forward(tR)                             # [1,9]
        yRh = self.forward(tRh)                            # [1,9]
        slopeR = (yR - yRh) / eps                           # [1,9]

        tn = t_norm.squeeze(-1)                            # [B]
        out = y0.clone()

        idxL = torch.nonzero(tn < -1.0, as_tuple=False).squeeze(-1)
        if idxL.numel() > 0:
            d = (tn[idxL] + 1.0).unsqueeze(-1)            # [BL,1]
            out[idxL, :] = yL.expand(idxL.numel(), -1) + slopeL.expand(idxL.numel(), -1) * d

        idxR = torch.nonzero(tn >  1.0, as_tuple=False).squeeze(-1)
        if idxR.numel() > 0:
            d = (tn[idxR] - 1.0).unsqueeze(-1)            # [BR,1]
            out[idxR, :] = yR.expand(idxR.numel(), -1) + slopeR.expand(idxR.numel(), -1) * d

        return out