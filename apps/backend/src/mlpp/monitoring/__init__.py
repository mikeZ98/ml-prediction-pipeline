"""Production monitoring: capture what the service scores, compare it with what it learned.

`payloads` is stdlib + pandas only and is imported by the serving path. `drift`
loads Evidently lazily (the `monitor` extra), so the API never pays for it.
"""
