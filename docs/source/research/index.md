# In the Literature

DiffeRT2d was first developed as a toolbox for my own research on
Differentiable Ray Tracing applied to Radio Propagation.

This page aims at listing usage of this library for scientific
researches.

:::{note}
While this currently only contains publications I collaborated
to, please contact me if you used DiffeRT2d for one of your scientific
projects, so I can add your publication (or else) to this list!
:::

## 2024

Publications in 2024.

### COST20120 Meeting in Helsinki

```{toctree}
:hidden:

../notebooks/cost20120_helsinki_model
```

At the COST Meeting held in Helsinkin, June 2024,
we will present a Machine Learning model that aims to reduce the computational
cost of Ray Tracing.
This model uses Differt2d for both (1) generating the training
data, and (2) constructing the actual ray paths from the sampled path candidates.

The model is completely open source and is detailed in
[this notebook](../notebooks/cost20120_helsinki_model.ipynb).

### Journal of Open Source Software

We are preparing a paper submission for the JOSS, for which the source files
are located in the
[`papers/joss`](https://github.com/jeertmans/DiffeRT2d/tree/main/papers/joss)
folder.

### EuCAP 2024

Our work,
*Fully Differentiable Ray Tracing via Discontinuity Smoothing for Radio Network Optimization*
{cite}`fully-eucap2024`,
utilizes this library as a framework to simulate Differentiable Ray Tracing
through smoothing, as done in the {mod}`differt2d.logic` module with
the use of
{func}`activation<differt2d.logic.activation>` functions.

This library's repository also contains the code to reproduce the plots
present in the paper, see the
[`papers/eucap2024`](https://github.com/jeertmans/DiffeRT2d/tree/main/papers/eucap2024)
folder.
