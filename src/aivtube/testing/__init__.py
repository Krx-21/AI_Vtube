"""Test support that ships with the package (ARCHITECTURE.md §10).

- ``aivtube.testing.fakes``: a fake for every adapter Protocol in ``aivtube.contracts``.
- ``aivtube.testing.contracts``: reusable contract suites that real implementations and fakes
  must both pass.
- ``aivtube.testing._conformance``: mypy-only Protocol assignments (fake and real classes); add
  a line there for every new implementation so signature drift fails CI.

Nothing here imports PortAudio, onnxruntime, sherpa-onnx or pytest.
"""
