# Manuscript

Compile from this directory with:

```bash
latexmk -pdf -interaction=nonstopmode -halt-on-error ieee-dc-template.tex
```

The main comparison uses NYUv2, KITTI, ETH3D, ScanNet, and DIODE. The two ablation
tables use NYUv2 and KITTI only. All eleven training configurations are mapped to
the tables in [the ablation guide](../config/ablation/README.md).

The manuscript remains a draft: ablation values are pending, and author-supplied
main-table values require verification against the final experiment protocol.
The generated PDF is included for review. Figure working files, backups,
reference-paper material, and LaTeX build files remain local.
