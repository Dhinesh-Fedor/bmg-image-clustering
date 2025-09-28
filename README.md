# BMP Image Clustering using PySpark MLlib

![Python](https://img.shields.io/badge/Python-3.x-blue.svg) ![Apache
Spark](https://img.shields.io/badge/Apache%20Spark-3.x%2B-orange.svg)

This project uses **PySpark** to perform unsupervised clustering on a
directory of BMP images. It demonstrates a complete ML pipeline: loading
unstructured image data, converting it into numerical feature vectors,
and grouping similar images using the **KMeans** algorithm.

The goal is to automatically identify and group visually similar images
without any manual labeling.

------------------------------------------------------------------------

### 📋 Table of Contents

1.  [Workflow Overview](#-workflow-overview)
2.  [Project Structure](#-project-structure)
3.  [Prerequisites](#-prerequisites)
4.  [How to Run](#-how-to-run)
5.  [Key Concepts & Insights](#-key-concepts--insights)

------------------------------------------------------------------------

### 🚀 Workflow Overview

The process transforms raw image files into meaningful clusters through
several key stages:

-   **1. Load Images**: The `Dataset/` directory is scanned, and all
    `.bmp` image files are loaded as raw binary data into a Spark
    DataFrame.
-   **2. Preprocess & Vectorize**: Each image's binary data is processed
    by a User-Defined Function (UDF). The UDF converts the image to
    **grayscale**, **resizes** it to a uniform 32x32 pixels, and
    **flattens** the 2D pixel grid into a single 1D array.
-   **3. Feature Extraction**: The pixel values in the 1D array are
    **normalized** (scaled to a range of 0.0 to 1.0) and converted into
    a Spark ML `Vector`. This 1024-dimensional vector becomes the
    feature set for the model.
-   **4. KMeans Clustering**: The KMeans algorithm analyzes the feature
    vectors, grouping images that are close to each other in
    1024-dimensional space (i.e., have a low Euclidean distance) into
    the same cluster.
-   **5. Analyze Results**: The final output assigns a cluster ID to
    each image, allowing you to see which images have been grouped
    together.

------------------------------------------------------------------------

### 📁 Project Structure

For the script to work correctly, your project folder should be
organized as follows:

    bmg-image-clustering/                                                                                                                      
         ├── Dataset/  
         │       ├── video01.mp4_000002.bmp 
         │       ├── video01.mp4_000018.bmp 
         │             └── ... (and so on) 
         ├── ProcessingImagesUsingPyspark.ipynb \# Your Jupyter Notebook 
         └── README.md \# This file

------------------------------------------------------------------------

### 🛠 Prerequisites

Ensure you have Python and the following libraries installed.

-   Python 3.x
-   PySpark 3.x or 4.x
-   NumPy
-   Pillow (PIL Fork)

You can install the dependencies using pip: \`\`\`bash pip install
pyspark numpy pillow


------------------------------------------------------------------------

### 💻 How to Run:

1)  Clone the Repository (Optional) If your project is on GitHub, clone
    it:

Bash

`git clone https://github.com/Dhinesh-Fedor/bmg-image-clustering.git`
 
`cd bmp-image-clustering`

2)  Add Your Images Place all the `.bmp image` files you want to analyze
    inside the `Dataset/` folder.

3)  Launch `Jupyter Notebook` In your terminal, navigate to the project's
    root folder and run:

4)  Execute the Code Open your notebook (.ipynb file) and run the cells
    sequentially.


------------------------------------------------------------------------

### 💡 Key Concepts & Insights 

1) Unsupervised Learning: KMeans is an unsupervised algorithm. It
finds inherent structures (clusters) in the data without needing any
pre-labeled examples.

2) Feature Engineering for Images: The process of converting an image into
a numerical vector (\[0.1, 0.5, ...\]) is a form of feature engineering.
It makes unstructured data understandable for a machine learning
algorithm.

3) High-Dimensional Space: Each 32x32 image becomes a point in a
1024-dimensional space. KMeans calculates the "distance" between these
points to determine similarity.

4) Scalability: Using PySpark allows this process to scale from a few
images on a local machine to millions of images on a distributed cluster
with minimal code changes.

5) Robustness: The try-except block in the UDF is critical for real-world
data pipelines, ensuring that a few corrupt files don't cause the entire
job to fail.


------------------------------------------------------------------------

### 📜 License 
This project is licensed under the MIT License. See the
LICENSE file for details.
