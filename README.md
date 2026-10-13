
# 🏴‍☠️ Luffy – AI Data Analyst

An AI-powered data analysis web application built with Python, designed to help users explore datasets, generate insights, visualize data, and experiment with machine learning.

🚀 **Live Demo:** [Try Luffy – AI Data Analyst](https://aidataanalyst-1.streamlit.app/)

## ✨ Features

- 📂 **Data Upload:** Upload datasets for analysis.
- 📊 **Data Visualization:** Explore data through interactive charts and visualizations.
- 🧹 **Data Cleaning:** Prepare datasets for analysis.
- 🤖 **AI-Powered Insights:** Generate insights from data using AI.
- 💬 **AI Chat Assistant:** Interact with Luffy, your AI data assistant.
- 📈 **Machine Learning:** Explore machine learning capabilities.
- 📑 **AI Reports:** Generate reports from your analysis.
- 🌐 **Multilingual Interface:** Access AI features in multiple languages.
- 🔐 **User Authentication:** Sign up and log in to the application.

## 🛠️ Technologies

- **Python** – Core programming language
- **Streamlit** – Web application framework
- **Pandas** – Data manipulation and analysis
- **Plotly** – Interactive data visualization
- **Scikit-learn** – Machine learning
- **Supabase / PostgreSQL** – Database and user authentication
- **Groq** – AI-powered features
- **ReportLab** – PDF report generation
- **Git & GitHub** – Version control and collaboration

## ⚙️ Getting Started

### 1. Clone the repository

```bash
git clone https://github.com/ZennourChaimae1/AI_Data_Analyst.git
cd AI_Data_Analyst
```

### 2. Create a virtual environment

```bash
python -m venv venv
```

Activate it:

**Windows**
```bash
venv\Scripts\activate
```

**macOS / Linux**
```bash
source venv/bin/activate
```

### 3. Install dependencies

```bash
pip install -r requirements.txt
```

### 4. Configure environment variables

Create a `.env` file in the project root and add the required credentials:

```env
GROQ_API_KEY=your_groq_api_key
SUPABASE_URL=your_supabase_url
SUPABASE_KEY=your_supabase_key
```

Replace the example values with your own credentials. Never commit real API keys or passwords to GitHub.

### 5. Run the application

```bash
streamlit run app.py
```

## 📁 Project Structure

```text
AI_Data_Analyst/
├── app.py
├── database.py
├── requirements.txt
├── .env
├── .gitignore
└── README.md
```

*The project structure above is illustrative. Update it to match the actual files in your repository.*

## 🔒 Security

- Keep API keys and database credentials private.
- Store secrets in environment variables or Streamlit Secrets.
- Protect user data and ensure that each user can only access their own information.

## 👩‍💻 Author

**Zennour Chaimae**

- GitHub: [@ZennourChaimae1](https://github.com/ZennourChaimae1)

## 🌟 Try the Application

Explore the live application:

👉 https://aidataanalyst-1.streamlit.app/



*Built with Python, Data Analytics, and AI.*

