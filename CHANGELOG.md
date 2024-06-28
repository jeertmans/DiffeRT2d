# Changelog

All notable changes to this project will be documented in this file.

The format is based on [Keep a Changelog](https://keepachangelog.com/en/1.0.0/)
and this project adheres to [Semantic Versioning](https://semver.org/spec/v2.0.0.html).

<!-- start changelog -->

(unreleased)=
## [Unreleased](https://github.com/jeertmans/DiffeRT2d/compare/v0.3.3...HEAD)

(v0.3.3)=
## [v0.3.3](https://github.com/jeertmans/DiffeRT2d/compare/v0.3.2...v0.3.3)

(v0.3.3-added)=
### Added

+ Added `Vertex` class for basic vertex diffraction.
  [#70](https://github.com/jeertmans/DiffeRT2d/pull/70)
+ Added `get_vertices` method to `Wall`.
  [#70](https://github.com/jeertmans/DiffeRT2d/pull/70)
+ Added `filter_objects` method to `Scene`.
  [#70](https://github.com/jeertmans/DiffeRT2d/pull/70)
+ Added `filter_objects` parameters to `Scene.all_path_candidates`
  and related methods.
  [#70](https://github.com/jeertmans/DiffeRT2d/pull/70)
+ Added a lower-level, cached, variant of `all_path_candidates`.
  [#70](https://github.com/jeertmans/DiffeRT2d/pull/70)

(v0.3.3-chore)=
### Chore

+ Fixed broken links and added test to check for links.
  [#68](https://github.com/jeertmans/DiffeRT2d/pull/68)
+ Enhanced the documentation homepage.
  [#77](https://github.com/jeertmans/DiffeRT2d/pull/77)

(v0.3.2)=
## [v0.3.2](https://github.com/jeertmans/DiffeRT2d/compare/v0.3.1...v0.3.2)

Small documentation enhancements.

(v0.3.1)=
## [v0.3.1](https://github.com/jeertmans/DiffeRT2d/compare/v0.3.0...v0.3.1)

(v0.3.1-added)=
### Added

+ Added utilities to rename scene transmitters and receivers.
  [#62](https://github.com/jeertmans/DiffeRT2d/pull/62)

(v0.3.1-changed)=
### Changed

+ Changed the types of indices (and path candidates) to
  allow signed integers, as a way to support negative indexing.
  [#62](https://github.com/jeertmans/DiffeRT2d/pull/62)

(v0.3.1-chore)=
### Chore

+ Added a detailed notebook about our ML model presented at
  the COST20120 INTERACT meeting, Helsinki, June 2024.
  [#62](https://github.com/jeertmans/DiffeRT2d/pull/62)

(v0.3.0)=
## [v0.3.0](https://github.com/jeertmans/DiffeRT2d/compare/v0.2.0...v0.3.0)

(v0.3.0-added)=
### Added

+ Path methods now accept `Point` in addition to `Float[Array, "2"]` for `tx`
  and `rx` arguments.
  [#59](https://github.com/jeertmans/DiffeRT2d/pull/59)

(v0.3.0-changed)=
### Changed

+ Changed `.point` and `.points` class attributes to `.xy` and `.xys`
  to emphasize that they are the raw x-y coordinates.
  This is a **breaking change**.
  [#59](https://github.com/jeertmans/DiffeRT2d/pull/59)
+ Changed `Scene` attributes to me immutable, to follow the principle
  of PyTrees.
  This is a **breaking change**.
  [#59](https://github.com/jeertmans/DiffeRT2d/pull/59)
+ Removed the `seed` argument for path methods and made the `key` argument
  mandatory when needed. The `key` and `kwargs` arguments are now present
  everywhere, but may be unused.
  This is a **breaking change**.
  [#59](https://github.com/jeertmans/DiffeRT2d/pull/59)

(v0.3.0-chore)=
### Chore

+ Changed to project to use modern tools like Rye, Ruff (discarding Black
  and others), `bump-my-version`, ReadTheDocs documentation hosting, better
  type hints, etc.
  This is a **breaking change** because some functions parameters changed
  (e.g., `minimize`), a type checker is used on many functions, and classes
  are now frozen dataclasses (because PyTrees).
  [#58](https://github.com/jeertmans/DiffeRT2d/pull/58)
+ Update the examples and lib to match changes in codebase and satisfy Pyright.
  [#59](https://github.com/jeertmans/DiffeRT2d/pull/59)
+ Reworked documentation to include notes on JAX, jaxtyping, and scientific
  publications.
  [#59](https://github.com/jeertmans/DiffeRT2d/pull/59)

(v0.2.0)=
## [v0.2.0](https://github.com/jeertmans/DiffeRT2d/compare/v0.1.0...v0.2.0)

(v0.2.0-added)=
### Added

+ Added three keyword arguments to `Scene.plot`, to make plotting transmitters,
  receivers or objects, an option.
  [#37](https://github.com/jeertmans/DiffeRT2d/pull/37)
+ Added a new `annotate_offset` argument to `Point.plot` to shift the text.
+ Added options to compute the gradient (or the value and the gradient) of
  accumulated functions from `Scene` class.

(v0.2.0-changed)=
### Changed

+ Renamed all occurrences of `emitter*` to `transmitter*`.
  This is a **breaking change** regarding some function names.
  [#43](https://github.com/jeertmans/DiffeRT2d/pull/43)
+ Accumulate functions from `Scene` class now return an iterator by default,
  which can be reduce with the `reduce_all` option.
  This is a **breaking change**.
+ Changed `function` parameter in `activation` to be a callable, not a `str`,
  so anyone can provide a custom activation function.
  This is a **breaking change**.
  [#45](https://github.com/jeertmans/DiffeRT2d/pull/45)
+ Changed the default value for `many` parameter to `1` when generating paths
  using `MPT` or `FPT`. Prior to that, `10` was used.
+ Renamed environ variable `JAX_ENABLE_APPROX` to `DISABLE_APPROX`.
  Setting this variable to any value will disable approximation.
  This is a **breaking change**.
  [#45](https://github.com/jeertmans/DiffeRT2d/pull/54)
+ Changed the default value of enabling (or not) approximation from `jax.config`
  (removed from their API) to `differt2d.logic.set_approx`.
  This is a **breaking change**.
  [#45](https://github.com/jeertmans/DiffeRT2d/pull/54)

(v0.2.0-chore)=
### Chore

+ Added benchmarks to better quantify performance changes.

(v0.2.0-fixed)=
### Fixed

+ Fixed how path candidates are computed, which actually removes the
  `get_visibility` matrix method and changes the output type of `all_path_candidates`
  to a list of arrays.
  [#48](https://github.com/jeertmans/DiffeRT2d/pull/48)

## [v0.1.0](https://github.com/jeertmans/DiffeRT2d/commits/v0.1.0)

This version added all basic features. This changelog will only document feature
changes in future versions.

<!-- end changelog -->
