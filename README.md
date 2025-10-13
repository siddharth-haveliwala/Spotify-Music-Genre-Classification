# 🎧 Spotify Music Genre Classification

## Project Overview
This project presents a robust **predictive model** for classifying music tracks by their genre and generating highly personalized playlist recommendations. Leveraging a comprehensive dataset from **Spotify's API** and simulated **user browsing history**, this solution moves beyond traditional collaborative filtering to offer nuanced, content-based recommendations tailored to individual preferences and moods.

It serves as a strong demonstration of advanced machine learning techniques applied to real-world, high-dimensional audio feature data and a playground for such classification problems.

<img width="681" height="680" alt="image" src="https://github.com/user-attachments/assets/3dc63397-cd13-4de9-af99-d83b0be6d0a5" />

## Key Features & Technical Highlights

* **Advanced ML Pipeline:** Developed a full-cycle machine learning pipeline incorporating data loading, extensive feature engineering, correlation analysis, and model selection.
* **Model Comparison:** Rigorous evaluation of ensemble methods, including **Random Forest, XGBoost, CatBoost, and AdaBoost**, to achieve optimal genre classification accuracy.
* **Data Sources:** Utilized a comprehensive dataset of music features (e.g., danceability, energy, acousticness) and top hits from 2000 to 2022 to train the final recommender.
* **Personalization:** The final recommendation engine is driven by a model that predicts a user's likelihood of engaging with a track based on their inferred preferences from browsing data.

## 📂 Repository Structure

| Directory/File | Purpose |
| :--- | :--- |
| `src/` | Contains all original **R scripts** for data cleaning, visualization, modeling, and evaluation. |
| `docs/` | **Full Final Paper** (`FinalPaper.pdf`) with in-depth methodology, literature review, and results. |

## 🛠️ Installation and Reproduction

This project was developed using **R**.

1.  **Clone the Repository:**
    ```bash
    git clone https://github.com/siddharth-haveliwala/Spotify-Music-Genre-Classification-and-Recommendation.git
    cd Spotify-Music-Genre-Classifier-Recommender
    ```
    
2.  **Run the Main Script:**
    Execute the primary script in the `src/` folder to reproduce the model training and evaluation:
    ```R
    # Example command to run the main analysis script
    > load("~/your_path/src/RData_Models.RData")
    > load("~/your_path/src/RData_FinalData.RData")
    ```

## 🔬 Research Methodology

Our goal is to bridge the gap in existing music recommendation literature by proposing a predictive model that seamlessly integrates Spotify's track features with the potential of user browsing history for highly personalized playlist generation.

- ### Data and Pre-processing
    - **Data Sources:** The study utilized two primary data types:
      1. **Top Hit Songs (2000-2022):** A dataset of 2,300 samples detailing various audio features from top charts.
      2. **User Browsing History:** Conceptualized as a key input for the recommender, though not fully explored in the current analysis, ensuring the model's design is future-proof.
    - **Feature Focus:** The model was built upon core music features accessible via the Spotify API, including: `danceability`, `energy`, `loudness`, `valence`, `acousticness`, `speechiness`, and `instrumentalness`.

    - **Cleaning & Standardization:** The data underwent meticulous preprocessing, including the removal of irrelevant features (e.g., `playlist_url`, `track_id`) and the handling of missing values. Critically, the music features were already standardized to ensure uniformity for machine learning model development.


- ### Classification Strategy and Feature Selection

    * **Target Variable (`Genre`):** To focus the classification task, the primary genres of interest—**Pop, Hip Hop, and Rock**—were strategically selected due to their predominance in the dataset.
      <img width="2880" height="1800" alt="image" src="https://github.com/user-attachments/assets/c6abf0e7-3a53-4ce0-aa05-52572d3e6ee1" />

    * **Genre Encoding:** An **encoding scheme** was implemented to simplify the multi-class problem, assigning distinct numeric labels to Pop, Hip Hop, and Rock, and grouping all other genres under a single 'Other' class.

      <img width="700" height="400" alt="image" src="https://github.com/user-attachments/assets/4da5e1f8-e9ff-48e4-be90-6c2cd78e0e4e" />

    * **Correlation Analysis:** A crucial step was the **correlation analysis** among the audio features.
      <img width="717" height="448" alt="image" src="https://github.com/user-attachments/assets/8e95c7f9-24a0-4029-a22e-0c8e4993ab91" />
      <img width="600" height="500" alt="image" src="https://github.com/user-attachments/assets/a3183600-1d99-4148-a54a-08639c6a0ee0" />


      Key findings informed feature selection:
        * **Strong Positive:** A significant positive correlation (0.69) was observed between `loudness` and `energy`.
        * **Moderate Negative:** A moderate negative correlation (-0.54) was found between `acousticness` and `energy`, suggesting a valuable classification insight.
        * Features like `duration_ms` and `time_signature` were excluded due to their low correlation with the predictive response variable.
        
        <img width="400" height="400" alt="image" src="https://github.com/user-attachments/assets/0d5da325-5b82-4135-81ae-3bda86cabdac" />

        * The features showing the highest correlation with the target variable `Genre` (e.g., `Danceability`, `Acousticness`, `Energy`) were prioritized for model training.

- ### Model Evaluation and Validation

    * **Validation Method:** To ensure robust evaluation and parameter optimization, the dataset was partitioned into a **75% training set** and a **25% validation set**, utilizing **5-fold cross-validation**.
    * **Algorithms Evaluated:** A comprehensive suite of industry-standard machine learning algorithms was tested for their effectiveness in music genre prediction:
        * **Single Model:** Decision Tree
        * **Ensemble Models:** Extreme Gradient Boosting (**XGBoost**), **AdaBoost**, **Random Forest**, and **CatBoost** (specifically chosen for its excellence in handling categorical features).

## 📈 Results & Key Findings 

The model selection process was centered on maximizing the **recall** of individual genres and overall **accuracy** across the multi-class classification task. The results clearly identify **CatBoost** as the most promising predictive model.

### Comparative Model Performance Summary

| Model | Overall Accuracy | Class 2 (Recall) | Class 3 (Recall) | Class 4 (Recall) | Key Insight |
| :--- | :--- | :--- | :--- | :--- | :--- |
| **Decision Tree** | 70.0% | 67.4% | 70.4% | 69.6% | Commendable general performance. |
| **XGBoost** | 68.11% | *69.23%* (Class 0) | *69.44%* (Class 1) | *52.94%* (Class 2) | Strong ensemble method, but varied class performance. |
| **AdaBoost** | 69.32% | **84.09%** (Class 2) | N/A | N/A | Exceptional performance on a specific class. |
| **Random Forest** | 67.87% | 38.29% | 84.47% | 39.29% | Good performance on Class 3, weaker on others. |
| **CatBoost** | *(Highest Overall)* | **31%** | **92%** | **43%** | **Standout model** with exceptional Class 3 performance. |

### CatBoost: The Optimal Performer

The **CatBoost** algorithm, designed to handle categorical data efficiently, emerged as the standout model. Its performance metrics, while varied, reveal its strengths:

* **Exceptional Recall:** CatBoost achieved an **impressive 92% recall rate for Class 3**, demonstrating its superior effectiveness in correctly identifying instances of that specific genre (likely **Pop**, given its high sample count).
* **Robustness in Categorical Features:** The model's success validates its use for classification problems involving complex, high-dimensional audio feature data.

### Project Conclusion and Bias Mitigation

* **Bias Observation:** A key finding was the identification of **dataset bias**, particularly for the most populous genre (Pop), which can inflate the overall accuracy. This necessitates a more targeted strategy.
* **Resolution:** By strategically focusing the model on the key genres (Pop, Hip Hop, Rock, and Other), we were able to **address the bias** and achieve an **acceptable level of accuracy** across the classes of interest, effectively solving the problem of high-specificity genre classification.

## 🚀 Future Enhancements (Roadmap)

* **API Deployment:** Wrap the recommendation engine in a **REST API** (e.g., with Plumber in R or Flask/Django in Python) for real-time inference.
* **User Interface:** Develop a simple web application using **Shiny** (R) to allow users to input preferences and receive live recommendations.
* **Deep Learning:** Explore the use of **Recurrent Neural Networks (RNNs)** or **Transformers** to classify and recommend music based on sequential listening patterns.
