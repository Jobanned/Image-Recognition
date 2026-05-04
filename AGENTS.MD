# AGENTS.md

This document defines the specialized "agents" or functional units within the `jobanned/image-recognition` project[cite: 1]. Each unit is responsible for a specific stage of the pipeline, from raw data acquisition to final classification and visual feedback[cite: 1].

---

## 1. Vision Perception Agent (VPA)
The **VPA** acts as the primary interface between the hardware (camera) and the processing logic[cite: 1]. Its goal is to ensure high-quality frames are available for analysis[cite: 1].

*   **Primary Tasks:** 
    *   Initialize and manage the camera stream via `app.py`[cite: 1].
    *   Normalize incoming image data to the required input size for the model[cite: 1].
*   **Dependencies:** OpenCV, `app.py`[cite: 1].

## 2. Landmark Detection Agent (LDA)
The **LDA** is a specialized spatial analyzer[cite: 1]. It focuses on identifying specific structural points within an image, such as hand joints, before high-level recognition occurs[cite: 1].

*   **Primary Tasks:** 
    *   Process frames using the `hand_landmarker.task` protocol[cite: 1].
    *   Map geometric coordinates for skeletal tracking[cite: 1].
*   **Assets Utilized:** `hand_landmarker.task`[cite: 1].

## 3. The "Monkey" Inference Agent (MIA)
The **MIA** is the decision-making core[cite: 1]. It interprets the processed spatial data and attempts to classify the content based on pre-trained logic[cite: 1].

*   **Operational States:**
    *   **Knowing State:** Activated when a positive identification is made with high confidence[cite: 1]. Represented by `monkeyknow.png`[cite: 1].
    *   **Thinking State:** Activated during active inference or when data is ambiguous[cite: 1]. Represented by `monkeythink.png`[cite: 1].
*   **Primary Tasks:** 
    *   Execute model inference[cite: 1].
    *   Trigger state-based visual feedback[cite: 1].

## 4. Interaction & Feedback Agent (IFA)
The **IFA** manages the user-facing output, translating the internal state of the MIA into a visual display for the end user[cite: 1].

*   **Primary Tasks:** 
    *   Overlay detection results onto the camera feed[cite: 1].
    *   Toggle between "Think" and "Know" visual assets in the UI[cite: 1].
*   **Dependencies:** Flask/Web Framework logic in `app.py`[cite: 1].

---

## Agent Workflow Pipeline

1.  **Vision Perception Agent** captures a raw frame[cite: 1].
2.  **Landmark Detection Agent** extracts spatial features (e.g., hand positions)[cite: 1].
3.  **Monkey Inference Agent** enters **Thinking** state (`monkeythink.png`) while calculating[cite: 1].
4.  **Monkey Inference Agent** reaches a conclusion and enters **Knowing** state (`monkeyknow.png`)[cite: 1].
5.  **Interaction & Feedback Agent** renders the final result to the user interface[cite: 1].