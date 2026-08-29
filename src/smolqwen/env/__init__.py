"""Environment layer: registry, live instances, tool schemas, verifier, worker pool.

Nothing here imports an LLM client, an HTTP client, or an API SDK. The layer's
whole job is to execute released Python against live state and score it, and a
network call in that path would make a reward depend on something outside the
episode.
"""

from __future__ import annotations
