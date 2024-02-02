# 🧭 MetaBrest Mission
> There is relevancy in capturing breast deformation in MRI scans since these are taken with the patient facing the floor, but surgery is performed the other way around, thus possibly paving the way for avoidable surgeon mistakes.

> **The goal is to create a biomechanical generative model to produce synthetic, high-fidelity and high-quality (even if using Super Resolution) breast MRI scans to then be fed to Deep Learning models. These should also be created conditionally and in a patient-specific manner so as to improve automated diagnosis research.**

# 🔍 Research Ideas
> **The usage of masks to condition the position and shape of tumours could be a biased approach, since basing these on previous cases of tumour-typology could be disregarding the information contained in the patient-specific input images.**

> Generated Images have slightly less contrast. Inter-Breast Muscle is also irregularly defined and often of an excessive width. Finally, generated pectorals do not match the anatomic extremity. All these discrepancies between real and generated can be measured and used as punishment losses, although this will require either manual labour or simple localization models.

# 📌 Step-by-Step

## Non-Conditional
- 3D Image Diffusion Model
    - Noise Distribution
        - ~~Gaussian Noise~~
        - Black-Out Diffusion
        - Gamma Noise
    - Architectural Modifications
        - Beta Scheduler
        - Position Embedding Layer
        - Spatial Attention Layer
- Comparative State-of-the-Art Models
    - 3D Generative Adversarial Network
    - 3D Normalizing Flow

## Conditional Generation
- Vector-Quantized Diffusion Model
- Image-to-Image Translation 3D Diffusion Model
- Adversarially Trained Conditional Diffusion Model (with High Discriminator Threshold)
- Physics-Informed Deformation-Based 3D Image Generative Model

# ⚖️ Patient-Specific Parameters

## Static & Image-Implicit
- Age
- Breast Density
- Breast Volume vs. Fibroglandular Tissue

## Dynamic
- Indispensable
    - ????
- Other
    - ????
