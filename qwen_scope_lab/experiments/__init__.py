"""Research experiments built on top of the core lab.

Each experiment is a self-contained pipeline that composes the lab's primitives
(steering, scoring, recipes) into a larger workflow. The first one,
``steering_distill``, compiles a validated runtime steer into ordinary
training/evaluation data — a bridge from interpretability to deployable
model improvement.
"""
