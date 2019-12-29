import numpy as np # linear algebra
import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)
import seaborn as sns
import matplotlib.pyplot as plt

#plotly

from plotly.offline import init_notebook_mode, iplot
#init_notebook_mode(connected=True)
import plotly.graph_objs as go
import plotly.figure_factory as ff
from IPython.display import HTML, Image
from plotly import tools
import folium
from folium import plugins
import squarify

from geopy.geocoders import Nominatim
from geopy.extra.rate_limiter import RateLimiter

renfe = pd.read_csv("renfe.csv")
for i in ['insert_date','start_date','end_date']:
    renfe[i] = pd.to_datetime(renfe[i])

renfe = renfe.fillna
renfe = renfe.dropna()
print(renfe)



