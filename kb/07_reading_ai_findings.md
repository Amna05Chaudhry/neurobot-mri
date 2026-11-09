# How to read NeuroBot's findings

Abnormality detected: the slice is flagged when the predicted abnormal region is larger than a small minimum size. Very small predicted regions are ignored because they are often noise.

Area %: the fraction of the 2-D image covered by the predicted mask. It is not a tumour volume; volume needs every slice of the 3-D scan.

Location: the hemisphere (left/right) and quadrant (anterior/posterior) of the centre of the mask on the image. NeuroBot assumes the standard radiological display convention, in which the patient's left appears on the right side of the image, and assumes the front of the head is at the top. If the image orientation is different, the reported location will be wrong.

Bounding box: the smallest rectangle around the predicted region, in pixels of the 256x256 image.

Confidence: the mean predicted probability inside the mask, and the maximum probability on the slice. High confidence does not mean the prediction is correct; models can be confidently wrong, especially on images unlike their training data.

What a positive or negative result means: a positive finding means the model saw FLAIR-bright tissue resembling lower-grade glioma on this slice. Many other conditions (stroke, demyelination, infection, post-treatment change) are also FLAIR-bright. A negative result on one slice does not rule out disease elsewhere in the brain.

Sources:
- Radiopaedia, Fluid attenuated inversion recovery: https://radiopaedia.org/articles/fluid-attenuated-inversion-recovery
- Radiopaedia, MRI sequences (overview): https://radiopaedia.org/articles/mri-sequences-overview
