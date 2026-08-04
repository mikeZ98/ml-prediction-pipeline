"""One module per PRD surface. Each exposes `render(...)` over already-loaded objects.

Panels never load artifacts themselves and never spell a filename — `loaders` owns
every read and every cache decision, and `manifest.filenames_for(ROLE_*)` owns every
name. Deliberately no re-exports.
"""

from __future__ import annotations
