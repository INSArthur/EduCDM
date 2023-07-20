# coding: utf-8
# 2021/4/23 @ tongshiwei

import numpy as np
import warnings
warnings.filterwarnings("error")

__all__ = ["irf", "irt3pl"]


def irf(theta, a, b, c, D=1.702, *, F=np):
    try :
        result = c + (1 - c) / (1 + F.exp(-D * a * (theta - b)))
    except RuntimeWarning :
        print([theta,a,b,c,D])
    return result


irt3pl = irf
