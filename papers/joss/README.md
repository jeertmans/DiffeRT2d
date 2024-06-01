# JOSS Paper

This folder contains the source files used to generate
the paper we submitted to the
[Journal of Open Source Software](https://joss.theoj.org/).

You can generate a draft version of the PDF with:

```bash
docker run --rm \
    --volume $PWD/papers/joss:/data \
    --user $(id -u):$(id -g) \
    --env JOURNAL=joss \
    openjournals/inara
```

The `$PWD/papers/joss` assumes that you are running
the command from the root directory of this repository.
