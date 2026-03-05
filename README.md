# Maize Yield Prediction — Uasin Gishu County, Kenya
**IBM SkillsBuild Data Analytics Bootcamp**

## Live Dashboard
[Open the interactive dashboard](https://maize-yieldprediction.streamlit.app)

## Project Summary
This project predicts annual maize yield (t/ha) in Uasin Gishu County, Kenya using seasonal weather, soil, and seed management data from 2012–2023.

## Key Findings
- Rainfall timing matters more than total volume
- Pre-planting conditions (Sep–Nov) shape the following harvest
- Soil acidity (pH 5.7) explains the structural yield gap of ~6.5 t/ha

## Model Performance
- Algorithm: Random Forest + SVR
- Validation: Leave-One-Out Cross-Validation (n=12)
- LOO RMSE: 0.364 t/ha (10% of county average)
- LOO R²: 0.304

## Data Sources
- Ministry of Agriculture & Livestock Development, Kenya
- NASA POWER Climate Data
- Lomurut (2014), Purdue University
- CIMMYT / Tegemeo Institute
