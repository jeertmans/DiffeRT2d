---
title: 'DiffeRT2d: A Differentiable Ray Tracing Python Framework for Radio Propagation'
theme: white
revealOptions:
  controls: false
  progress: false
  slideNumber: 'h/v'
  transition: 'slide'
---

<img alt="DiffeRT2d's logo" src="logo.png" width="200">

**DiffeRT2d**: A Differentiable Ray Tracing Python Framework for Radio Propagation

*Jérome Eertmans - 2024-06-16*

---

## Second slide

> Best quote ever.

Note: speaker notes FTW!

---

## Examples

* [Exploring Metasurfaces and More](#/example1)
* [Network optimization](#/example2)
* [Machine Learning](#/example3)

---

<!-- .slide: id="example1" -->

<img alt="RIS" src="ris_power_map.pdf">

![RIS](ris_power_map.pdf)

RIS model that always reflects with an angle of 45°.

----

```python [1|2-5|6|7|8-14]
scene = Scene.square_scene()
ris = RIS(
    xys=jnp.array([[0.5, 0.3], [0.5, 0.7]]),
    phi=jnp.pi / 4,
)
scene = scene.add_objects(ris)
X, Y = scene.grid(n=300)
P = scene.accumulate_on_receivers_grid_over_paths(
    X, Y,
    fun=received_power,
    path_cls=MinPath,
    order=1,
    ...,
)
```

---

<!-- .slide: id="example2" -->

MDR

---

<!-- .slide: id="example3" -->

PTDR

---

![Demo with Qt interface](demo.gif)

---

## Thanks for you attention!

<img alt="QR code" src="qrcode.svg" width="500">
