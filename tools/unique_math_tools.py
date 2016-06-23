#!/usr/bin/env python
# -*- coding:utf-8 -*-

import numpy as np

def sigmoid(z):
    return 1.0 / (1.0 + np.exp(np.where(z < -709, 709, -z)))
