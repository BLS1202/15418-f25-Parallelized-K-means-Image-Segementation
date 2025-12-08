# Project Title

[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](https://opensource.org/licenses/MIT)

A brief one-sentence description of what your project does. This should be catchy and informative.

## Description

15418 Fall 2025 project. We implemented K-means clustering image segmentation in CUDA on GPU and in OpenMP on the CPU and compared the performance of non-optimized versions with these two parallelized versions. Our implementation demonstrated significant speedups. We also parallelized the initialization step of K-means++ in CUDA, and implemented a CUDA based image renderer that can run CUDA and OpenMP implementations of K-means clustering segmentation on image for every frame.

![Screenshot of the project](path/to/your/screenshot.png)

## Table of Contents

- [Installation](#installation)
- [Usage](#usage)
- [Features](#features)
- [Technologies Used](#technologies-used)
- [Contributing](#contributing)
- [License](#license)
- [Contact](#contact)
- [Acknowledgments](#acknowledgments)

## Installation

Provide clear, step-by-step instructions on how to get a development environment running.

1.  **Prerequisites**

    List any software, libraries, or tools that need to be installed before your project can be set up.

    ```bash
    npm install npm@latest -g
    ```

2.  **Clone the repository**

    ```bash
    git clone https://github.com/your-username/your-repo.git
    cd your-repo
    ```

3.  **Install dependencies**

    ```bash
    npm install
    ```
    or for Python projects:
    ```bash
    pip install -r requirements.txt
    ```

4.  **Set up environment variables**

    Create a `.env` file and add the necessary environment variables. You can provide a `.env.example` file as a template.

    ```
    API_KEY="YOUR_API_KEY"
    DATABASE_URL="YOUR_DATABASE_URL"
    ```

## Usage

Explain how to use your project. Provide code examples, command-line instructions, or a description of how to interact with the UI.

To run the application in development mode:

```bash
npm run dev
