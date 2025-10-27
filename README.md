MIT License — see `LICENSE`

![GhanaSegNet](https://img.shields.io/badge/project-GhanaSegNet-blue)

# GhanaSegNet

GhanaSegNet — a master's-level research codebase for semantic segmentation of Ghanaian food imagery.

---

## Abstract

> Food recognition and automated nutritional assessment remain complex challenges in sub-Saharan Africa, particularly in Ghana, where diverse traditional meals pose unique computational difficulties for artificial intelligence systems. The limited representation of African cuisines in existing food image datasets has created a significant gap in global food computing research. Current models are largely trained on Western-centric datasets such as Food101 and Recipe1M, resulting in algorithmic bias and poor performance when applied to culturally specific dishes. This thesis addresses this challenge through the development of GhanaSegNet, a hybrid convolutional–transformer-based semantic segmentation model specifically designed for Ghanaian food imagery. The study introduces a multi-stage transfer learning framework that progressively adapts pretrained models from general visual domains to culturally relevant food contexts. The research employs the FRANI dataset, comprising 1,141 annotated images of common Ghanaian dishes categorized into six semantic classes. A comprehensive data preprocessing and augmentation pipeline was developed to enhance model robustness, simulate real-world presentation styles, and mitigate class imbalance.
>
> The GhanaSegNet architecture integrates the efficiency of convolutional encoders with the contextual reasoning power of transformer bottlenecks, enabling the model to capture both local and global visual dependencies. A composite loss function combining Dice and Boundary losses was employed to address class imbalance and improve segmentation precision, particularly along object boundaries. The model was trained using a structured, multi-resolution training protocol and evaluated on a held-out validation split using mean Intersection over Union (mIoU) as the primary performance metric. Experimental results demonstrated that GhanaSegNet achieved an average validation mIoU of 24.47%, performing competitively with the more complex DeepLabV3+ model (25.44%) while maintaining a significantly smaller parameter count. The model also exhibited superior boundary delineation and stable convergence across epochs, highlighting the effectiveness of its hybrid design and composite loss formulation.
>
> The findings of this study provide evidence that culturally tailored computer vision models can achieve high segmentation accuracy and efficiency even within limited-resource settings. The multi-stage transfer learning approach and boundary-aware training framework demonstrate a practical pathway for developing inclusive and contextually relevant AI models in the African setting. The research contributes to addressing algorithmic bias in food computing and establishes a foundation for future work on mobile deployment and real-world nutritional assessment. Overall, GhanaSegNet presents a significant step toward the integration of culturally responsive artificial intelligence systems for health and nutrition applications in Ghana and beyond.

**Keywords:** semantic segmentation · GhanaSegNet · food recognition · transfer learning · computer vision

---

*For developer instructions and reproducing experiments, see `docs/DEVELOPER_GUIDE.md` (recommended).*
