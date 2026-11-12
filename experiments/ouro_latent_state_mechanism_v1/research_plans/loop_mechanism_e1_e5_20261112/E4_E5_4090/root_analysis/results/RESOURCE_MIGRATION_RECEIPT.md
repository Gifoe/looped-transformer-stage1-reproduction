# Resource migration receipt

This chat owns E4/E5 on the AutoDL 4090. Sol owns the Python environment and experiment executables; Astra owns scientific orders/reviews. The other active chat owns E1–E3.

- Existing local checkpoint: SHA256 `58872a72616c736595b8b7662079c5b12c5a162ec16eae94f21c348dfa9885af`, 2,869,336,434 bytes. Eight SFTP streams sustained about 11.35 MiB/s and the assembled remote copy passed the same SHA256. Existing incomplete copies were preserved with separate names. No model parameters were edited.
- Model source SHA256 `c5c68fbb368ce2909c257ae2afc50719be8c91539333d3295e19312c4316f413`; config SHA256 `ce9cc13da41591b8b4deca053d7dfee06424c0228628ee862ea86d725bc163f3`, matching audited source provenance.
- The original server path was read-only except for new, isolated metadata export script/artifacts. A 39 MB preservation ZIP was checked but only contained 448 removed metadata files. The 2,590-question portable export was constructed from those plus 2,142 original metadata files, using per-split coverage-linked hashes. Compressed export: 1,967,984 bytes, SHA256 `3989bde25b0e80207b88e24431c6a866dc57c3adde23fd98df68e0a08418bd2f`.
- AutoDL academic proxy availability was verified against its official documentation and actual GitHub access. It did not make every large package fast. Existing source-server SFTP and HF range probes were slow, so neither was allowed to complete a redundant checkpoint download.
- CUDA wheels were fetched locally from official PyPI using exact versions and SHA256 values verified against PyPI metadata, then staged with parallel SFTP and checked again remotely. The first 1.03 GB (cuDNN and cuBLAS) completed in about 100 seconds. Pip installation remains Sol-owned and uses local wheelhouse files.

Official references: https://api.autodl.com/docs/network_turbo/ and https://docs.pytorch.org/get-started/previous-versions/ . Actual transfer/download scripts and JSON receipts are retained under exec/ and results/. No authentication secrets are stored in these files.
