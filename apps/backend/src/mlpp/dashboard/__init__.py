"""Streamlit presentation layer over a trained session.

Sits on the **TensorFlow-owning** side of the module split: `loaders` imports
`mlpp.predict`, which imports Keras. Nothing in `config`, `data`, `preprocess`,
`metrics`, `session`, `errors` or `importance` may import this package, or the fast
suite stops being TensorFlow-free.

It renders; it does not compute. Anything that calculates belongs in a normal `mlpp`
module, importable and testable without Streamlit — which is why permutation
importance lives in `mlpp.importance` rather than in a panel.

Deliberately no re-exports: importing this package must not drag Keras in.
"""

from __future__ import annotations
