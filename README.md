![Python 3.5](https://img.shields.io/badge/python-3.5-blue.svg)

# ML-Metrics

A simple way to get automated threshold analysis using Streamlit.

![Muestra](https://raw.githubusercontent.com/JAVI897/ML-Metrics/master/muestra.gif "Muestra")

## Requirements

* python 3.7 version
* streamlit 0.48.1 version
* plotly 4.2.1 version
* numpy
* pandas

## Run

```sh
pip install streamlit plotly numpy pandas
git clone https://github.com/JAVI897/ML-Metrics.git
# cd into the project root folder
cd ML-metrics
streamlit run app.py
```

## Getting Started

Clone or download the repository.

Place all of your predictions and your ground truths (saved as numpy arrays) in the data folder so that your data folder look something like this.

```
.
├──data
|   ├──prediction_1.npy
|   ├──test_1.npy
```

