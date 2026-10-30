# Fire and Smoke Detection Reproductions

Research notes, code, experiment outputs, and group meeting presentations for two fire and smoke detection projects.

## Projects

### APConv / AEL-YOLO

Located in [`APConv_Finaldata_复现/`](APConv_Finaldata_复现/). This project reproduces the authors' APConv-based detector on the Finaldata fire and smoke dataset. It includes model code, training and validation scripts, experiment notes, metrics, visualizations, and presentation materials.

### YOLO-GN

Located in [`YOLO_GN_复现/`](YOLO_GN_复现/). This project contains the YOLO-GN model changes, training and evaluation scripts, experiment notes, result visualizations, and group meeting materials.

## Data and large files

Datasets, virtual environments, model checkpoints, local server bundles, and large archives are intentionally kept outside Git. On a Mac, clone this private repository, transfer the datasets from your authorized remote server over SSH or SFTP, then update the local dataset paths in the YAML configuration files before training. Dataset directories are ignored by Git, so transferred data stays local. Small result tables, plots, example images used in presentations, PDFs, and slides are included.

## Citation and upstream code

See each project README for paper references, upstream repositories, licenses, setup notes, and reproduction details. This repository records independent reproduction work; reported results may use settings that differ from the original papers.
