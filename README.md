# 🚀 Genspark Hyper Personalization Recommendation

## 📌 Table of Contents
- [Introduction](#introduction)
- [Demo](#demo)
- [Inspiration](#inspiration)
- [What It Does](#what-it-does)
- [How We Built It](#how-we-built-it)
- [Challenges We Faced](#challenges-we-faced)
- [How to Run](#how-to-run)
- [Tech Stack](#tech-stack)
- [Team](#team)

----

## 🎯 Introduction
Tailor highly personalized experiences and financial services to online banking customers that cater to their unique preferences. 
Enhance hyper personalization by analysing customer profile, social media activity, purchase history, sentiment data and demographic details.


## 🎥 Demo
📹 [Video Demo](https://drive.google.com/file/d/10ADNtvQYxwn-M5IgftG2cnRWTmu9xzCa/view?usp=sharing) 
🖼️ Screenshots:

### Bank products curated through LLM and recommendation engine
![image](https://github.com/user-attachments/assets/a198f9be-84aa-4d7d-9393-828610f1c6ff)

### Partnered merchants offers curated through LLM and recommendation engine
![image](https://github.com/user-attachments/assets/59303595-0e4a-4f3f-9d60-12e4974af543)

### Alert push notifications by Gen AI during active hours of the users
![image](https://github.com/user-attachments/assets/139fffd5-f754-4602-85e3-af3aca766676)



## 💡 Inspiration
Many recommendation systems fail to capture dynamic user preferences in real-time, leading to irrelevant suggestions. Our goal was to build a system that adapts continuously.

## ⚙️ What It Does
Using AI & real-time data to tailor financial services to individual users
- Personalized Banking Product Recommendations (Loans, credit cards, investment plans)
- Targeted Merchant Offers & Discounts 
- Customized services /alerts for budgeting/saving
- Customized Insights  for Business insights & data-driven recommendations. 

## 🛠️ How We Built It
- Graph Neural Networks(GNN) capturing User-Product Relations
- Wide & Deep Learning for Hybrid Recommendations
- Reinforcement Learning(RL) for continuous Learning
- Real-time Recommendation Pipeline


## 🚧 Challenges We Faced
- Data collection issues due to Customer data being distributed across multiple systems
- Token limit constraints affecting LLM usage
- Increased loading times caused by the computations demands of LLMs


## 🏃 How to Run
1. Clone the repository  
   ```sh
   git clone https://github.com/ewfx/aidhp-gen-spark
   ```
2. Install dependencies for frontend
   ```sh
   npm install  
   ```
3. Run the frontend
   ```sh
   npm start  
   ```
4. Install the dependencies for recommendation engine server
   ```sh
   pip install 
   ```
5. Run the recommendation engine server
   ```sh
   python recommendation.py
   ```
6. Install the dependencies for  Gen AI server
   ```sh
   pip install
   ```
7. Run the dependencies for  Gen AI server
   ```sh
   python app.py
   ```

## 🏗️ Tech Stack
- 🔹 Frontend: React/Vite/Tailwind
- 🔹 Backend: Flask/FastAPI
- 🔹 LLM Model: Llama-3.2/Deepseek R1
- 🔹 Data Embedding: Bert/GNN
- 🔹 Model: wide&deep model
- 🔹 Databse: MongoDB
- 🔹 Other: Ollama,tensorflow,Faker

## 👥 Team
- **Vedant Singh** - [GitHub](https://github.com/vedant-11) | [LinkedIn](https://www.linkedin.com/in/vedant-singh-a7145020a/)
- **Harsh Thakur** - [GitHub](https://github.com/HarshThakur-08) | [LinkedIn](https://www.linkedin.com/in/harsh-thakur-b18b7920a/)
- **Aryan Kharbanda** - [GitHub](https://github.com/aryankharbanda) | [LinkedIn](https://www.linkedin.com/in/aryan-kharbanda-a6552a206/)
- **Tanishq Kakkar** - [GitHub](https://github.com/tanishq1308) | [LinkedIn](https://www.linkedin.com/in/tanishq-kakkar-663100201)
- **Sandhya Adabala** - [GitHub](https://github.com/sandhyaadabala) | [LinkedIn](https://www.linkedin.com/in/sandhya-adabala-7b44534/)
