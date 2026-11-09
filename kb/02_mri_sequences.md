# MRI sequences used for brain tumours

T1-weighted images show anatomy well; fluid (CSF) is dark. Most lower-grade gliomas are low signal (dark) on T1.

T1-weighted post-contrast (gadolinium) images show areas where the blood-brain barrier is disrupted. Many lower-grade gliomas do not enhance. New or increasing enhancement in a known low-grade glioma can be a sign of transformation to a higher grade and should be reported to the treating team.

T2-weighted images show fluid and oedema as bright. Lower-grade gliomas are typically T2-hyperintense (bright).

FLAIR (fluid-attenuated inversion recovery) is a T2-weighted sequence in which the signal from free fluid such as CSF is suppressed. Tumour and oedema stay bright while the ventricles turn dark, which makes the extent of a non-enhancing glioma easier to see. The "T2-FLAIR mismatch sign" (bright on T2, relatively dark centre on FLAIR) is a highly specific imaging marker of IDH-mutant, 1p/19q non-codeleted astrocytoma.

Advanced sequences: diffusion-weighted imaging (DWI/ADC) reflects cellularity; perfusion imaging (DSC/DCE, rCBV) helps estimate tumour grade; MR spectroscopy measures metabolites such as choline and NAA, and 2-hydroxyglutarate in IDH-mutant tumours.

NeuroBot's model only sees a 3-channel image made of pre-contrast, FLAIR and post-contrast slices, and its training masks were drawn on FLAIR. It therefore segments FLAIR abnormality, which includes tumour and may include surrounding oedema. It cannot tell tumour from oedema and cannot determine grade.

Sources:
- Radiopaedia, Fluid attenuated inversion recovery: https://radiopaedia.org/articles/fluid-attenuated-inversion-recovery
- Radiopaedia, T2-FLAIR mismatch sign: https://radiopaedia.org/articles/t2-flair-mismatch-sign
- Radiopaedia, MRI sequences (overview): https://radiopaedia.org/articles/mri-sequences-overview
- Buda M, Saha A, Mazurowski MA. Computers in Biology and Medicine 2019. https://doi.org/10.1016/j.compbiomed.2019.05.002
