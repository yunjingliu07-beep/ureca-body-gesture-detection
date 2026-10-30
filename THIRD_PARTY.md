# Third-party code and models

## Gamebody

The `Gamebody/` directory contains source code, key maps, and documentation
adapted from [everythingishacked/Gamebody](https://github.com/everythingishacked/Gamebody).
Its original GNU General Public License v3 is preserved in
[Gamebody/LICENSE](Gamebody/LICENSE).

Local setup changes use the author's macOS keyboard fork and add a compatible
dependency file and setup notes. Gamebody is included as ordinary source files,
not as a submodule; cloning this project includes the code directly. Upstream
Git history and demo media are not included.

The keyboard dependency is pinned to
[`everythingishacked/keyboard` at 83bcc0421165e9ad610343525b7cfc13d5145a95](https://github.com/everythingishacked/keyboard/tree/83bcc0421165e9ad610343525b7cfc13d5145a95).
Its source and license are maintained in that repository.

## MediaPipe

The extractor uses the MediaPipe Python package and Google's official model
bundles. Model binaries are not committed here; download URLs and checksums
are documented in [models/README.md](models/README.md).
