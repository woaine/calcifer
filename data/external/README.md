# Facial and Oral Temperature Data (v1.0.0)

This directory contains raw external dataset files derived from the publicly available study:

> Wang, Q., Zhou, Y., Ghassemi, P., Chenna, D., Chen, M., Casamento, J., Pfefer, J., & McBride, D. (2023). _Facial and oral temperature data from a large set of human subject volunteers (version 1.0.0)_. PhysioNet. [https://doi.org/10.13026/3bhc-9065](https://doi.org/10.13026/3bhc-9065)

and its associated publication:

> Wang, Q., Zhou, Y., Ghassemi, P., McBride, D., Casamento, J. P., & Pfefer, T. J. (2022). _Infrared Thermography for Measuring Elevated Body Temperature: Clinical Accuracy, Calibration, and Evaluation_. Sensors, 22, 215. [https://doi.org/10.3390/s22010215](https://doi.org/10.3390/s22010215)

This dataset is curated to evaluate the clinical performance of infrared thermographs (IRTs) for measuring elevated body temperature. In our study, we leverage this dataset to develop a non-invasive core body temperature prediction model using a multilayer perceptron (MLP) architecture and data acquired via long wavelength infrared (LWIR) cameras.

---

## Overview

The dataset includes thermal and visible images, oral temperature measurements, sensor readings (ambient temperature and humidity), and accompanying demographic metadata. Data collection involves two evaluated IRTs:

- **IRT-1 (FLIR camera)**
- **IRT-2 (ICI camera)**

Subjects’ oral temperatures and facial thermal images are recorded under controlled ambient conditions (grouped into two ranges: 20.0–24.0°C and 24.0–29.0°C). Facial temperature variables (26 per subject) are extracted using a free-form deformation method to register visible and infrared images. This data collection protocol maximizes clinical accuracy and reproducibility for evaluating performance in febrile screening and related public health applications.

---

## Data Collection and Processing

- **Imaging Data:**  
  Thermal images are captured using the two IRT systems. Visible images (from a webcam) serve as auxiliary data for accurate facial region registration and variable extraction.

- **Measurement Protocol:**  
  For each subject, oral temperature is recorded twice in monitor mode (preferred for its accuracy over fast mode). Facial images yield 26 temperature variables representing different facial regions, extracted using free-form deformation-based registration.

- **Data Integrity and Cleaning:**  
  Data cleaning excludes incomplete records or those affected by motion artifacts. The final dataset includes records from 1020 subjects (IRT-1) and 1009 subjects (IRT-2). All identifiers are de-identified in compliance with HIPAA (e.g., age is reported as age group, dates are coded).

---

## File Structure

The following files are included in this directory:

- **FLIR_group1.csv**  
  Captured by the FLIR camera under ambient conditions between 20.0°C and 24.0°C.
- **FLIR_group2.csv**  
  Captured by the FLIR camera under ambient conditions between 24.0°C and 29.0°C.
- **FLIR_groups1and2.csv**  
  A combined dataset for FLIR (Group 1 and Group 2).
- **ICI_group1.csv**  
  Captured by the ICI camera under ambient conditions between 20.0°C and 24.0°C.
- **ICI_group2.csv**  
  Captured by the ICI camera under ambient conditions between 24.0°C and 29.0°C.
- **ICI_groups1and2.csv**  
  A combined dataset for ICI (Group 1 and Group 2).
- **LICENSE.txt**  
  Provides licensing details; released under the Creative Commons Zero 1.0 Universal Public Domain Dedication.
- **SHA256SUMS.txt**  
  Contains checksums for verifying file integrity.
- **\_Figure1.png**  
  Illustrative figure from the original study.
- **\_Table1.pdf** and **\_Table2.pdf**  
  Describe the temperature variables and associated metadata.

---

## Usage Notes

Researchers can use this dataset to:

- Develop and evaluate non-invasive temperature estimation methods that combine facial thermal imaging with environmental sensor data.
- Assess the clinical accuracy and calibration of IRT systems in febrile screening scenarios.
- Analyze correlations between facial temperature patterns and external or demographic variables.

**Limitations:**

- The subject pool is predominantly young (95% under 30 years old), limiting generalizability to older populations.
- All data are collected under controlled ambient conditions; extrapolations to extreme environments require caution.

---

## License

This dataset is released under the Creative Commons Zero 1.0 Universal (CC0 1.0) license. See [LICENSE.txt](LICENSE.txt) for details.

---

## Acknowledgements

This dataset is developed with support from the U.S. Food and Drug Administration’s Medical Countermeasures Initiative (MCMi) Regulatory Science Program and the Center for Devices and Radiological Health. We thank the University Health Center at the University of Maryland, College Park, and all collaborators who contribute to the data collection and study implementation.

---

_This dataset complies with HIPAA guidelines and reflects rigorous ethical and de-identification standards to ensure subject privacy._
