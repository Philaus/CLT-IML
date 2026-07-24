# Contributing

Thank you for helping improve CLT-IML. Contributions that clarify data
provenance, improve reproducibility, fix platform-specific paths, or correct a
documented issue are particularly welcome.

## Reporting a problem

Open a focused issue and include:

- the script and command that were run;
- the working directory and any renamed/staged input files;
- the operating system and Python, PyTorch, CUDA, and MATLAB versions as
  applicable;
- CPU/GPU information;
- the complete error message or a minimal example; and
- whether the input is from this repository or a new CLT campaign.

Do not attach confidential CLT inputs, credentials, licensed third-party code,
or unpublished simulation data unless you are authorized to redistribute them.

## Pull requests

1. Keep changes scoped to one purpose.
2. Preserve the physical meaning, units, normalization, and provenance of data
   columns. Document any deliberate schema change.
3. Do not replace reference weights or datasets without updating checksums,
   metrics, and release notes.
4. Prefer configurable paths for new code. If changing a historical script,
   retain enough documentation to reproduce the paper snapshot.
5. Test the affected entry point from a clean working directory and describe
   the test in the pull request.
6. Avoid committing caches, local environments, credentials, or large generated
   outputs unrelated to the change.

By contributing, you confirm that you have the right to submit the code or data
and agree that its inclusion and licensing are subject to approval by Zhejiang
University as the software copyright owner.
