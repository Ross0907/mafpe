# Private Repository Policy

This working tree is treated as a private research repository and release source.

- Private GitHub release pushes should include accelerator RTL, reproducibility scripts, fixed-point memory fixtures, validation reports, Vivado reports, and generated PDF/PNG debug artifacts.
- Private GitHub release pushes should not include Python virtual environments, downloaded datasets, transient Vivado project directories, training checkpoints unless explicitly needed, or unrelated local files.
- `run_neuraedge_private_full.ps1` runs local export, Vivado simulation, Vivado implementation, validation, and report generation before packaging.
- Generated canonical reports are stored under `cifar10_neuromorphic\outputs_spikingformer_head_accelerator`.
- Large external environments and datasets stay local under `external_research` and `datasets`.
- The GitHub remote must be private before any accelerator release package is uploaded.

Current canonical local command:

```powershell
.\run_neuraedge_private_full.ps1
```
