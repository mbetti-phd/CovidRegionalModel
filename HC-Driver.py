import numpy as np
import matplotlib.pyplot as plt
from scipy.integrate import odeint
from lmfit import minimize, Parameters, Parameter, report_fit, fit_report
from scipy.integrate import odeint
import pandas as pd
import datetime
import os
import json
import csv
import random

import uncertainties.unumpy as unp
import uncertainties as unc

import HCbuildIncidence as hcb
import HCODEfit as hcode
import HCsummarize as hcs

hcb.main()
hcode.main()
hcs.main()