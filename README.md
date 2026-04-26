---
title: Multi Disaster Prediction
emoji: 🌪️
colorFrom: red
colorTo: yellow
sdk: docker
app_file: app.py
pinned: false
---

# Multi-Disaster CNN Ensemble

A deep learning ensemble model that predicts disaster types from images.

## Classes
- Cyclone
- Earthquake
- Flood
- Wildfire

## Model
Uses an ensemble of EfficientNetV2, ConvNeXt, and DenseNet201 backbones with a dense meta-classifier.