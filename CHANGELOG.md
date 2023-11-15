# Changelog

All notable changes to this project will be documented in this file.

The format is based on [Keep a Changelog](https://keepachangelog.com/en/1.0.0/)
and this project adheres to [Semantic Versioning](https://semver.org/spec/v2.0.0.html).

<!-- start changelog -->

## [Unreleased](https://github.com/jeertmans/DiffeRT2d/compare/v0.1.0...HEAD)

### Added

+ Added three keyword arguments to `Scene.plot`, to make plotting transmitters,
  receivers or objects, an option.
  [#37](https://github.com/jeertmans/DiffeRT2d/pull/37)
+ Added a new `annotate_offset` argument to `Point.plot` to shift the text.
+ Added options to compute the gradient (or the value and the gradient) of
  accumulated functions from `Scene` class.

### Changed

+ Renamed all occurences of `emitter*` to `transmitter*`.
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

### Chore

+ Added benchmarks to better quantify performance changes.

### Fixed

+ Fixed how path candidates are computed, which actually removes the
  `get_visibility` matrix method and changes the output type of `all_path_candidates`
  to a list of arrays.
  [#48](https://github.com/jeertmans/DiffeRT2d/pull/458

## [v0.1.0](https://github.com/jeertmans/DiffeRT2d/commits/v0.1.0)

This version added all basic features. This changelog will only document feature
changes in future versions.

<!-- end changelog -->
