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

Ray Tracing (RT) is arguably one of the most prevalent methodologies in the field of radio propagation modeling. However, the access to RT software is often constrained by its closed-source nature, licensing costs, or the requirement of high-performance computing resources. While this is typically acceptable for large-scale applications, it can present significant limitations for researchers who require more flexibility in their approach, while working on more simple use cases. We present DiffeRT2d, a 2D Open Source differentiable ray tracer that addresses the aforementioned gaps. DiffeRT2d employs the power of JAX [@jax2018github] to provide a simple, fast, and differentiable solution. Our library can be utilized to model complex objects, such as reconfigurable intelligent surfaces, or to solve optimization problems that require tracing the paths between one or more pairs of nodes. Moreover, DiffeRT2d adheres to numerous high-quality Open Source standards, including automated testing, documented code and library, and Python type-hinting.

# Statement of Need

In the domain of radio propagation modeling, a significant portion of the RT tools available to researchers are either closed-source or locked behind commercial licenses. This restricts accessibility, limits customization, and impedes collaborative advances in the field. Among the limited Open Source alternatives, tools such as PyLayers and Open3D fall short in offering the capability to easily differentiate code with respect to various parameters. This limitation presents a substantial challenge for tasks involving network optimization, where the ability to efficiently compute gradients is crucial. To our knowledge, SionnaRT [@sionnart] is one of the few radio propagation-oriented ray tracers that incorporates a differentiable framework, leveraging TensorFlow [@tensorflow] to enable differentiation. Despite its capabilities, SionnaRT's complexity can be a barrier for researchers seeking a straightforward solution for fundamental studies in RT applied to radio propagation. Moreover, we believe that the research is lacking a simple-to-use and highly-interpretable RT framework.

DiffeRT2d addresses these shortcomings by providing a comprehensive, Open Source, and easily accessible framework specifically designed for 2D RT. It integrates seamlessly with Python, ensuring ease of use while maintaining robust functionality. By leveraging JAX for automatic differentiation, DiffeRT2d simplifies the process of parameter tuning and optimization, making it an invaluable tool for both academic research and practical applications in wireless communications. This framework democratizes access to advanced RT capabilities, thereby fostering innovation and facilitating rigorous exploration in the field.

# Easy to Use Commitment

# Usage Examples

## Exploring Metasurfaces and More

The first motivation behind using an object-oriented paradigm is the ability
to create custom subclasses to implement different behavior for a given object.
This is the case for metasurfaces: those objects are usually.



## Network optimization

Thanks to [@deepmind2020jax]

We have used it in [@eertmans2024eucap]...

## Machine Learning

In another work we present at this COST meeting, we have developed

In DiffeRT2d, every single class is a PyTree... Thanks to [@kidger2021equinox]

# Stability and releases

Type checking is provided by [@jaxtyping2024github]

# Statement of Need

## Target Audience

## Comparison with Similar Tools

# Acknowledgments

# References
