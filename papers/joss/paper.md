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

# Summary

![DiffeRT2d' logo.\label{fig:logo}](logo.png)

Ray Tracing is arguably one of the most prevalent methodologies in the field of radio propagation modeling. However, the access to Ray Tracing software is often constrained by its closed-source nature, licensing costs, or the requirement of high-performance computing resources. While this is typically acceptable for large-scale applications, it can present significant limitations for researchers who require more flexibility in their approach, while working on more simple use cases. We present DiffeRT2d, a 2D open-source differentiable ray tracer that addresses the aforementioned gaps. DiffeRT2d employs the power of JAX [@jax2018github] to provide a simple, fast, and differentiable solution. Our library can be utilized to model complex objects, such as reconfigurable intelligent surfaces, or to solve optimization problems that require tracing the paths between one or more pairs of nodes. Moreover, DiffeRT2d adheres to numerous high-quality open-source standards, including automated testing, documented code and library, and Python type-hinting.

# Statement of Need

# Easy to Use Commitment

# Usage Examples

## Network optimization

Thanks to [@deepmind2020jax]

We have used it in [@eertmans2024eucap]...

## Machine Learning

In DiffeRT2d, every single class is a PyTree... Thanks to [@kidger2021equinox]

# Stability and releases

Type checking is provided by [@jaxtyping2024github]

# Statement of Need

## Target Audience

## Comparison with Similar Tools

# Acknowledgments

# References
