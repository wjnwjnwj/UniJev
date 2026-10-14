"""VideoJev: a standalone Jev-style decision model for video understanding.

Single forward pass over (video frames + question + candidates); a shared
decision head scores every candidate directly — no answer-token generation.
"""
MODEL_NAME = "VideoJev"
__version__ = "0.1.0"
