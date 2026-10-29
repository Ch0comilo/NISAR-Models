# Notebooks

## Public, portable Earthdata notebook

Open [NISAR_Water_Try_Your_Data.ipynb](NISAR_Water_Try_Your_Data.ipynb) in Jupyter
or [Colab](https://colab.research.google.com/github/Ch0comilo/NISAR-Models/blob/main/nisar-water/notebooks/NISAR_Water_Try_Your_Data.ipynb).
Each user enters their own Bearer token in a hidden prompt. It performs bounded
standalone acquisition, inference, temporal comparisons and a local API check.
The published notebook has no saved outputs or credentials. See
[the setup guide](../docs/notebook-earthdata.md).

## Original workspace notebooks

The original research workspace retains these executed notebooks locally:

- `NISAR_Water_Improved_Classifier.ipynb`: latest two improvement experiments,
  all five fresh round5 test images, inference and actual API calls.
- `NISAR_Water_Round3_Comparison.ipynb`: historical HGB/U-Net/OPERA comparison.
- `NISAR_Water_Exploration.ipynb`: earlier eight-image exploration.

They use local research data/results that are intentionally excluded from Git.
Their large executed files are also excluded, so a minimal fresh clone does
not advertise them as independently runnable. The original notebooks remain
unchanged on the development computer; `launch-notebook.ps1` is local research
tooling for that workspace.

For a fresh clone, use the README's offline smoke check and
`examples/predict_hdf5.py` with your own scientific observation. In a notebook
kernel using this environment, call `nisar_water.pipeline.predict_dataset(pila,
...)` as shown in `docs/testing-new-data.md`.
