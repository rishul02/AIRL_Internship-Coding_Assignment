# AIRL_Internship-Coding_Assignment
This repository contains solutions for the AIRL lab internship coding assignment.

        q1.ipynb : Vision Transformer (ViT) on CIFAR-10
        q2.ipynb : Text-driven Image Segmentation with SAM 2

How to Run (Colab)

          Open q1.ipynb or q2.ipynb directly in Google Colab.
          Execute all cells from top to bottom.
          All required installations are present at the top of each notebook.
          Ensure the runtime type is set to T4 GPU (Runtime > Change runtime type > select T4 GPU).

1. Vision Transformer (ViT) on CIFAR-10
   
  Objective: Develop and train a Vision Transformer from scratch using PyTorch, targeting the
             best possible test accuracy on the CIFAR-10 dataset.

  Requirements: Google Colab with GPU (Runtime → Change runtime type → GPU)

  Results:

            Metric         ValueTest
            Accuracy        83.97%
            Training Time   ~2-3 hours (T4 GPU)
   
  Model Config:
   
            pythonPatch Size: 4x4 (gives 64 patches from 32x32 images)
            Embedding Dim: 256
            Transformer Blocks: 6
            Attention Heads: 8
            Batch Size: 128
            Epochs: 200
            Learning Rate: 0.0003

 2.  Text-Driven Image Segmentation with SAM 2
    
   Objective: Implement a pipeline for segmenting objects from text prompts, leveraging modern detection (Grounding DINO / OWL-ViT) and SAM 2.

   Uses two models working together:
   
         GroundingDINO (or OWL-ViT fallback) - converts text to bounding boxes
         SAM 2 - converts boxes to pixel-perfect masks

How to Run

  Requirements: Google Colab with GPU
    
    1. Open q2.ipynb in Colab
    2. Enable GPU (Runtime → Change runtime type → GPU)
    3. Run cells in order
    4. First run takes ~5-8 min (downloads models)
    5. After that, ~30 seconds per image

  Pipeline

      "dog" → GroundingDINO → [bounding boxes] → SAM 2 → [segmentation masks]

  Models Used

      GroundingDINO (primary):      
              Open-vocabulary object detection
              ~700MB checkpoint
              Best accuracy but installation can fail
      
      OWL-ViT (automatic fallback):     
              Backup detection model
              Easier to install (via HuggingFace)
              Slightly lower accuracy but more reliable
      
      SAM 2.1 Hiera-Large:      
              Segment Anything Model 2
              ~850MB checkpoint
              Handles both images and videos
              Insanely good quality masks

 
  Technical Stack

      Both Projects:     
      Python 3.10+
      PyTorch 2.0+
      CUDA (for GPU acceleration)
      Google Colab (free GPU)
      
      Q1 Specific:      
      torchvision (CIFAR-10 + transforms)
      Custom ViT implementation
      
      Q2 Specific:      
      GroundingDINO / OWL-ViT (detection)
      SAM 2 (segmentation)
      Hydra (config management)
      OpenCV (video processing)
