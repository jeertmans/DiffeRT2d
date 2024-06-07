---
title: 'DiffeRT2d: A Differentiable Ray Tracing Python Framework for Radio Propagation'
tags:
  - radio propagation
  - channel modeling
  - ray tracing
  - differentiable
  - framework
  - Python
authors:
  - name: Jérome Eertmans
    orcid: 0000-0002-5579-5360
    affiliation: 1
  - name: Claude Oestges
    orcid: 0000-0002-0902-4565
    affiliation: 1
  - name: Laurent Jacques
    orcid: 0000-0002-6261-0328
    affiliation: 1
affiliations:
 - name: ICTEAM, UCLouvain, Belgium
   index: 1
date: 7 June 2024
bibliography: paper.bib
---

![DiffeRT2d' logo.\label{fig:logo}](logo.png){ width="30%" }

# Summary

Ray Tracing (RT) is arguably one of the most prevalent methodologies
in the field of radio propagation modeling. However, the access to RT
software is often constrained by its closed-source nature, licensing costs,
or the requirement of high-performance computing resources.
While this is typically acceptable for large-scale applications,
it can present significant limitations for researchers who require
more flexibility in their approach, while working on more simple use cases.
We present DiffeRT2d, a 2D Open Source differentiable ray tracer
that addresses the aforementioned gaps.
DiffeRT2d employs the power of JAX [@jax2018github] to provide a simple,
fast, and differentiable solution. Our library can be utilized to
model complex objects, such as reconfigurable intelligent surfaces,
or to solve optimization problems that require tracing
the paths between one or more pairs of nodes.
Moreover, DiffeRT2d adheres to numerous high-quality Open Source standards, including
automated testing, documented code and library, and Python type-hinting.

# Statement of Need

In the domain of radio propagation modeling,
a significant portion of the RT tools available to researchers
are either closed-source or locked behind commercial licenses.
This restricts accessibility, limits customization,
and impedes collaborative advances in the field.
Among the limited Open Source alternatives,
tools such as PyLayers [@pylayers] and Opal [@opal]
fall short in offering the capability to easily differentiate
code with respect to various parameters.
This limitation presents a substantial challenge for tasks
involving network optimization, where the ability to efficiently
compute gradients is crucial.
To our knowledge, SionnaRT [@sionnart] is one of the few
radio propagation-oriented ray tracers that incorporates a
differentiable framework, leveraging TensorFlow [@tensorflow]
to enable differentiation.
Despite its capabilities, SionnaRT's complexity can be a barrier
for researchers seeking a straightforward solution for fundamental
studies in RT applied to radio propagation.
Moreover, we believe that the research is lacking a simple-to-use
and highly interpretable RT framework.

DiffeRT2d addresses these shortcomings by providing a comprehensive,
Open Source, and easily accessible framework specifically designed for 2D RT.
It integrates seamlessly with Python,
ensuring ease of use while maintaining robust functionality.
By leveraging JAX for automatic differentiation,
DiffeRT2d simplifies the process of parameter tuning and optimization,
making it an invaluable tool for both academic research and practical
applications in wireless communications.

Moreover, in contrast to the majority of other RT tools,
DiffeRT2d is capable of supporting a multitude of RT methods.
These include the image method [@imagemethod],
path minimization based on Fermat's principle [@fpt],
and the Min-Path-Tracing method (MPT) [@mpt-eucap2023].
Each of these methods represents a distinct compromise between speed and
the type of interaction that can be simulated, such as reflection or diffraction.

DiffeRT2d democratizes access to advanced RT capabilities,
thereby fostering innovation and facilitating rigorous exploration in the field.

# Easy to Use Commitment

DiffeRT2d is a 2D RT toolbox that aims to provide
a comprehensive solution for path tracing,
while avoiding the need to compute electromagnetic (EM) fields.
Consequently, we provide a rough approximation of the received power,
which ignores the local phase of the wave, to allow the user to focus
on higher-level concepts, such as the number of multipath components, angle of arrival.
As an object-oriented package with curated default values,
constructing a basic RT scenario can be performed in a minimal amount of lines of
code while keeping the code extremely expressive.

Moreover, DiffeRT2d is designed to maximize its compatibility with the JAX ecosystem.
It provides JAX-compatible objects, which are immutable, differentiable,
and jit-in-time compilable. This enables users to leverage the full capabilities
of other JAX-related libraries, such as Optax [@deepmind2020jax]
for optimization problems or Equinox [@kidger2021equinox] for Machine Learning (ML).

# Usage Examples

The documentation contains
[an example gallery](https://differt2d.readthedocs.io/latest/examples_gallery/),
as well as numerous other usage examples disseminated throughout the
application programming interface (API) documentation.

In the following sections, we will highlight a few of
the most attractive usages of DiffeRT2d.

## Exploring Metasurfaces and More

The primary rationale for employing an object-oriented paradigm
is the capacity to generate custom subclasses, enabling the implementation
of novel characteristics for a given object. This is exemplified by metasurfaces,
which typically exhibit a deviation from the conventional law of specular
reflection. Consequently, a distinct procedure must be employed for their treatment.

Using MPT,
that is one of the path tracing methods implemented in DiffeRT2d,
we can easily accommodate those object,
thanks to the object-oriented structure of the code.
We also provide a very simple reflecting intelligent surface (RIS) to this end.

![The following figure depicts a coverage map for single-reflection paths (i.e., no line-of-sight) in a scene containing a RIS. It is evident that the RIS reflects rays with an angle of 45°. The minor noise observed around the edges is attributed to convergence issues with the MPT method, which can be mitigated by increasing the number of minimization steps.\label{fig:rispowermap}](ris_power_map.pdf){ width="70%" }

\autoref{fig:rispowermap} can be reproduced[^1] with the following code:

[^1]: The code to plot the coverage map has been removed for clarity.

```python
import jax
import jax.numpy as jnp

from differt2d.geometry import RIS, MinPath
from differt2d.scene import Scene
from differt2d.utils import received_power

scene = Scene.square_scene()
ris = RIS(
    xys=jnp.array([[0.5, 0.3], [0.5, 0.7]]),
    phi=jnp.pi / 4,
)
scene = scene.add_objects(ris)

key = jax.random.PRNGKey(1234)  # Needed to start MPT minimization
X, Y = scene.grid(n=300)

P = scene.accumulate_on_receivers_grid_over_paths(
    X,
    Y,
    fun=received_power,
    path_cls=MinPath,
    order=1,
    reduce_all=True,
    key=key,
)
```

## Network optimization

In a previous work, we presented a smoothing technique [@eertmans2024eucap]
that makes RT differentiable everywhere. The aforementioned technique
is available throughout DiffeRT2d via an optional `approx`
(for *approximation*) parameter, or via a global config variable.

\autoref{fig:opt} shows how we used the Adam optimizer [@adam],
provided by the Optax library, to successfully solve some optimization problem.

![Illustration of the different iterations converging towards the maximum of the objective function, see [@eertmans2024eucap] for all details.\label{fig:opt}](optimize_steps.pdf){ width="100%" }

## Machine Learning

In a separate study presented at this COST meeting,
we developed an ML model that learns how to sample path candidates
to accelerate RT in general.

The model and its training were implemented using the DiffeRT2d library,
and a detailed notebook is available
[online](https://eertmans.be/r/cost20120-helsinki).

# Stability and releases

A significant amount of effort has been invested in the documentation and testing of our code. All public functions are annotated, primarily through the use of the jaxtyping library [@jaxtyping2024github], which enables static type checking. Furthermore, we aim to maintain a code coverage metric of 100%.

Our project adheres to semantic versioning,
and we document all significant changes in a changelog file.

## Target Audience

The intended audience for this software is researchers engaged
in the field of radio propagation who are interested in simulating
relatively simple scenarios. In such cases, the ease of use, flexibility,
and interpretability of the software are of greater importance than
performing city-scale simulations or computing
electromagnetic fields[^2] with high accuracy.

[^2]: While this is currently not part of our API, we do not omit the possibility
    to include more complex EM routines in the future.

# Acknowledgments

We would like to acknowledge the work from all contributors of
the JAX ecosystem, especially Patrick Kidger for the jaxtyping
and Equinox packages.

# References
