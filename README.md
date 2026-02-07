# Relative Mode


<!--conda activate relative_mode-->

<!--conda activate myenv--->

This package contains Python code for calculating *relative mode* from
an audio signal. *Relative mode* refers to the degree between how major
or minor does the segment of music sound at a given time. It is based on
a classic key-finding algorithm (Krumhansl-Schmuckler, 1990) and
extracts the pitch-class information using chromagrams. The relative
mode is calculated as the difference between the strongest major key and
the strongest minor key. Relative mode can vary from -1.0 (clearly in
minor) to + 1.0 (clearly in major) and gives a value between these
extremes for the whole excerpt. Alternatively the algorithm can provide
the output for each window of analysis (segments of 3 seconds as a
default).

The algorithm and how it is evaluated is fully documented in a
manuscript titled “Major-minorness in Tonal music – Evaluation of
Relative Mode Estimation using Expert Ratings and Audio-Based
Key-finding Principles” by Tuomas Eerola and Michael Schutz (*Psychology
of Music*, in press). Key-finding Principles” by Tuomas Eerola and
Michael Schutz (*Psychology of Music*,
[2025](https://doi.org/10.1177/03057356251326065)).

### Libraries

``` python
import librosa
import librosa.display
import matplotlib
import numpy as np
from matplotlib import pyplot as plt
import pandas as pd
```

### Load package using pip

``` python
pip install relative_mode
```

Make function calls explicit for the subsequent analyses.

``` python
from src.relative_mode import Tonal_Fragment
from src.relative_mode import relative_mode
from src.relative_mode import RME_across_time
```

### Load a music example

Recording of J.S. Bach’s *C Major Prelude* (WTC Book I) (an extract).

``` python
filename = 'data/Bach_1_Gould_0_Major_bachGould1971.wav'
y, sr = librosa.load(filename)
plt.figure(figsize=(9,2.5))
librosa.display.waveshow(y, sr = sr)
plt.show()
```

<div id="fig-waveform">

![](README_files/figure-commonmark/fig-waveform-output-1.png)

Figure 1: Waveform of the C Major Prelude.

</div>

### Estimate relative mode

Here we don’t specify any parameters but just run `relative_mode` using
the default parameters.

``` python
RM, RM_segments = relative_mode(y = y, sr = sr)
print(round(RM['tondeltamax'][0],3))
```

    0.258

The value of `0.258` could be called “moderately in major”. Value closer
to 0 would indicate no clear tendency for major or minor and any value
below -0.30 would suggest clearly in minor key.

The relative mode can be computed with a different options. You can
alter key profile (e.g. `krumhansl`, `albrecht` (default), `aarden`, or
`bellman`), similarity metrics (`pearson`, `cosine` (default), or
`euclidean`), chromatype from `CENS` to `CQT`. There are also some
alternative outputs of the measure.

Here’s a variant analysis using a different distance measure and chroma
type:

``` python
RM2, RM2_segments = relative_mode(y = y, sr = sr, profile = 'simple', distance = 'pearson', chromatype = 'CQT')
print(RM2)
```

       tonmaxmaj  tonmaxmin  tondeltamax
    0   0.688322   0.640134     0.144563

This outputs the `tondeltamax` value of `0.145`, which is the relative
mode with these parameters. The extra outputs refer to the highest
correlation coefficient with the major (`tonmaxmaj`) and minor
(`tonmaxmin`). Note that the distance metrics have different scales so
the outputs have been rescaled to be more easily comparable.

### Estimate relative mode across the excerpt

The second output provides a relative mode value for each window of the
analysis. The segment timing reflects the `hoplen` argument. Here we
also remove the percussive sounds with an extra parameter
(`remove_percussive=True`):

``` python
RM, RM_segments = relative_mode(y = y, sr = sr, winlen = 3.5, hoplen = 1.5, remove_percussive=True)
print(RM_segments)
```

        onset  tonmaxmaj  tonmaxmin   tonkey  tondeltamax
    0     0.0   0.895319   0.730949  C major     1.068404
    1     1.5   0.944695   0.805112  C major     0.907290
    2     3.0   0.921238   0.838886  C major     0.535289
    3     4.5   0.766527   0.850092  D minor    -0.543171
    4     6.0   0.876043   0.834032  G major     0.273075
    5     7.5   0.855947   0.757587  G major     0.639340
    6     9.0   0.927221   0.858836  G major     0.444506
    7    10.5   0.977284   0.850480  C major     0.824227
    8    12.0   0.878142   0.710265  C major     1.091195
    9    13.5   0.876234   0.803292  C major     0.474120
    10   15.0   0.807938   0.873190  A minor    -0.424143
    11   16.5   0.784958   0.905604  A minor    -0.784197
    12   18.0   0.774552   0.893590  A minor    -0.773744
    13   19.5   0.786032   0.799719  D minor    -0.088968
    14   21.0   0.907874   0.857099  D major     0.330038
    15   22.5   0.901390   0.771279  G major     0.845724
    16   24.0   0.913086   0.787238  G major     0.818011
    17   25.5   0.907057   0.806591  G major     0.653024
    18   27.0   0.819221   0.881004  E minor    -0.401589
    19   28.5   0.887434   0.874749  C major     0.082456

One can also visualise the relative mode across time. In this case there
is a cubic interpolation to make the lines between the windows appear
smooth, but one can alter this interpolation parameter.

``` python
fig, RM3 = RME_across_time(filename = filename, winlen = 2, hoplen = 2, cropfirst = 0, croplast = 15, chromatype = 'CENS', profile = 'albrecht', distance = 'cosine', plot = True, interpolation='cubic')
fig
plt.show()
```

<div id="fig-continuous">

![](README_files/figure-commonmark/fig-continuous-output-1.png)

Figure 2: Relative mode across time.

</div>

# Extras

## Weights to normalise the output across distance metrics

The `tondeltamax` output depends on the distance metric used. To
normalise close to between -1 and +1 for an easier use of the algorithm,
a weight is assigned to the raw delta value. These weights were
empirically derived by creating all possible 3 to 5-note chords and
calculating the relative mode with the available distance metrics. For
cosine distance metric, the weight is `6.5`, for pearson correlation,
`10.0`, and for euclidean distance, `3.0`. The purpose is to keep the
output more easily understandable
`(max major corr. - max minor corr.) * weigth`.

## Alternative analyses

In the article (Eerola & Schutz,
[2025](https://doi.org/10.1177/03057356251326065)), we assess various
parameters of the model (key profiles, distance measures, alternative
formulations of the model) in Experiment 1. We also examine what could
explain the variations in model success across recordings used in
Experiment 3. Here we briefly report these alternative explorations.

### Experiment 1: Alternative analyses

The model compares the difference between highest maximum major key
strength and the maximum minor key strength. We also have two
alternative formulations of the model, one that utilises comparison with
the parallel minor and another one relying on the relative minor.

The parallel minor key of the major key received lower correlation with
the expert ratings (*r* = 0.698) than the actual model (*r* = 0.840).
The second alternative relies on the relative minor key of the major
key. This alternative received a lower correlation (*r* = 0.766) with
the expert ratings compared to the proposed model. For this reason, we
did not pursue these two alternatives further.

We also run alternative formulations of window length (1 to 5 seconds)
and overlap (0 to 75% overlap) which did not provide substantially
better fit with the data. Finally, the way of summarizing the RME across
the analysis windows with the mean values did not appear to be
significantly different from the taking median of the predictions within
the analysis windows (*r* = 0.785).

### Experiment 3: Alternative analyses

To identify the consistent noise factors in the RME analysis from audio,
we extracted dynamics, several timbral descriptors (brightness, spectral
centroid, spectral flux, rms, roughness) and tempo descriptors for each
excerpt using [Essentia](https://essentia.upf.edu), and added these as
additional predictors to the regression with RME model predicting the
expert ratings. However, no single audio descriptor could contribute
significantly (more than 2 % of the variance accounted) to the model
that already had a highly successful predictor (RME) within it. A more
extensive analysis of the potential additional considerations would
benefit from a larger set of materials and from systematic alterations
of the most plausible variations of these factors.

### Improvements to the implementation

`Version 0.0.4`, 7 February 2026

- `RME_across_time` accepts interpolation parameter to control the way
  output is interpolated across the analysis windows. `cubic` is the
  default, but `linear` and `none` are possible as well.

- The output of the segments has now explicit time code (onset time in
  seconds).

- The parameters `cropfirst` and `cropfirst` now work for
  `RME_across_time`.

- The output of the `RM_across_time` has now `tonmaxmaj` and `tonmaxmin`
  output and the numeric output is unaffected by interpolation.

- A new option `remove_percussive` has been added to remove percussive
  noise (using `Librosa`’s median filtering solution, see
  `librosa.decompose.hpss`) from the signal. Filtering is set to `False`
  by default.

# References

Eerola, T. & Schutz, M. (2025). Major-minorness in Tonal music –
Evaluation of Relative Mode Estimation using Expert Ratings and
Audio-Based Key-finding Principles. *Psychology of Music, 0(0)*.
<https://doi.org/10.1177/03057356251326065>
