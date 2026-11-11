"""Last-step variable contributions, independently max-normalized."""
import torch
import torch.nn.functional as F

def variable_scores(x, reconstruction, beta=0.8):
    if not 0 <= beta <= 1:
        raise ValueError("beta must lie in [0,1]")
    p = (reconstruction - x).square()
    s = torch.zeros_like(p)
    if x.shape[1] > 1:
        s[:, 1:] = (torch.diff(reconstruction, dim=1)-torch.diff(x, dim=1)).square()
    def normalize(v):
        maximum = v.amax(dim=-1, keepdim=True)
        return torch.where(maximum > 1e-12, v/maximum.clamp_min(1e-12), torch.zeros_like(v))
    return beta*normalize(p[:, -1])+(1-beta)*normalize(s[:, -1])

def rank_event(window_scores, contributions, starts, event_start, event_end, window_size=30):
    """Event and window bounds are half-open row intervals; earliest maximum wins."""
    eligible=[i for i,start in enumerate(starts) if start < event_end and start+window_size > event_start]
    if not eligible:
        raise ValueError("No window overlaps the event")
    peak=max(eligible,key=lambda i:float(window_scores[i]))
    order=torch.argsort(contributions[peak],descending=True,stable=True)
    return peak,order
