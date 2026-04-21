# GPU Mapping

Shardon uses canonical GPU identities in config and scheduling.

## Identity Sources

- UUID when available
- PCI bus ID as a secondary stable key
- Observed probe output for validation and diagnostics

The scheduler and state files refer to configured GPU IDs such as `gpu0`, not backend-local device indices. A provider maps observed hardware data back into those stable IDs.

