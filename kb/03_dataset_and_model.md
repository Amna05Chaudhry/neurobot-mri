# About the NeuroBot dataset and model

NeuroBot was trained on the LGG Segmentation Dataset (Buda et al., 2019, known on Kaggle as "kaggle_3m"). It contains brain MR images from 110 patients with lower-grade glioma in The Cancer Genome Atlas (TCGA-LGG) collection, obtained through The Cancer Imaging Archive (TCIA). Each slice is a 256x256 RGB image built from three sequences (pre-contrast, FLAIR, post-contrast) with a manually drawn FLAIR abnormality mask. About a third of the slices contain tumour.

The model is a U-Net with a VGG-19 encoder pretrained on ImageNet. It outputs a per-pixel probability of FLAIR abnormality. The train/validation/test split is done by patient, so no patient's slices appear in more than one split. Results are reported on held-out test patients with the Dice coefficient and Intersection-over-Union (IoU); both range from 0 (no overlap) to 1 (perfect overlap).

Limitations: the model sees one 2-D slice at a time with no 3-D context. It was trained on a single public dataset of lower-grade gliomas only, and has not been validated on other scanners, other tumour types, post-operative brains or healthy volunteers. Its output is a research demonstration and must not be used for diagnosis.

Sources:
- Buda M, Saha A, Mazurowski MA. Association of genomic subtypes of lower-grade gliomas with shape features automatically extracted by a deep learning algorithm. Computers in Biology and Medicine 2019;109:218-225. https://doi.org/10.1016/j.compbiomed.2019.05.002
- Mazurowski MA et al. Radiogenomics of lower-grade glioma. Journal of Neuro-Oncology 2017. https://doi.org/10.1007/s11060-017-2420-1
- TCIA, TCGA-LGG collection: https://www.cancerimagingarchive.net/collection/tcga-lgg/
