# 🎧 Spotify Music Genre Classification & Recommendation Engine

## Project Overview
This project presents a robust **predictive model** for classifying music tracks by their genre and generating highly personalized playlist recommendations. Leveraging a comprehensive dataset from **Spotify's API** and simulated **user browsing history**, this solution moves beyond traditional collaborative filtering to offer nuanced, content-based recommendations tailored to individual preferences and moods.

It serves as a strong demonstration of advanced machine learning techniques applied to real-world, high-dimensional audio feature data and a playground for such classification problems.

## Key Features & Technical Highlights

* **Advanced ML Pipeline:** Developed a full-cycle machine learning pipeline incorporating data loading, extensive feature engineering, correlation analysis, and model selection.
* **Model Comparison:** Rigorous evaluation of ensemble methods, including **Random Forest, XGBoost, CatBoost, and AdaBoost**, to achieve optimal genre classification accuracy.
* **Data Sources:** Utilized a comprehensive dataset of music features (e.g., danceability, energy, acousticness) and top hits from 2000 to 2022 to train the final recommender.
* **Personalization:** The final recommendation engine is driven by a model that predicts a user's likelihood of engaging with a track based on their inferred preferences from browsing data.

## 📂 Repository Structure

| Directory/File | Purpose |
| :--- | :--- |
| `src/` | Contains all original **R scripts** for data cleaning, visualization, modeling, and evaluation. |
| `data/` | Contains the **final, pre-processed R data object** (`RData_FinalData.RData`). |
| `models/` | Contains the **trained classification and recommendation models** (`RData_Models.RData`). |
| `docs/` | **Full Final Paper** (`FinalPaper.pdf`) with in-depth methodology, literature review, and results. |
| `requirements.txt` | Lists all necessary R packages and their versions for full reproducibility. |

## 🛠️ Installation and Reproduction

This project was developed using **R**.

1.  **Clone the Repository:**
    ```bash
    git clone [https://github.com/YourUsername/Spotify-Music-Genre-Classifier-Recommender.git](https://github.com/YourUsername/Spotify-Music-Genre-Classifier-Recommender.git)
    cd Spotify-Music-Genre-Classifier-Recommender
    ```

2.  **Install Dependencies (R):**
    Open R/RStudio and run the following command to install all required packages listed in `requirements.txt`:
    ```R
    # Assuming your requirements.txt is adapted for R, or use this command to check dependencies
    source("src/dependency_checker.R") # A script you would write to install packages from a list
    ```
    *(Note: You will need to create the list of R packages used and put it in a file like `requirements.txt` or a similar R-friendly format.)*

3.  **Run the Main Script:**
    Execute the primary script in the `src/` folder to reproduce the model training and evaluation:
    ```R
    # Example command to run the main analysis script
    source("src/03_model_training_and_recommendation.R")
    ```

## 📈 Results & Key Findings

*(Here, you would add a high-level summary of your best model's performance—e.g., "The **CatBoost model** achieved an **F1-score of 92.5%** in genre classification, proving highly effective for the content-based filtering approach.")*

## 🚀 Future Enhancements (Roadmap)

* **API Deployment:** Wrap the recommendation engine in a **REST API** (e.g., with Plumber in R or Flask/Django in Python) for real-time inference.
* **User Interface:** Develop a simple web application using **Shiny** (R) to allow users to input preferences and receive live recommendations.
* **Deep Learning:** Explore the use of **Recurrent Neural Networks (RNNs)** or **Transformers** to classify and recommend music based on sequential listening patterns.
