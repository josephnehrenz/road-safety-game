# 🚗 Road Safety Game - Pick the Safer Road

An interactive web application that tests your intuition about road safety using machine learning. Built for the Kaggle & Stack Overflow Code Scientist Challenge.

[![Streamlit App](https://static.streamlit.io/badges/streamlit_badge_black_white.svg)](https://road-safety-game.streamlit.app/)

## 🎮 About the Game

Test your road safety knowledge! The app presents you with two random road scenarios and challenges you to pick which one is safer. After you guess, it reveals the actual risk scores predicted by a CatBoost machine learning model trained on real road safety data.

**Example Scenario:**
- **Road 1**: 3-lane highway, curvature 0.35, nighttime, rainy
- **Road 2**: 2-lane rural road, curvature 0.75, daylight, clear weather

Which would you choose?

## 🛠️ Technical Details

### Machine Learning
- **Model**: CatBoost Regressor
- **Features**: Road type, lanes, curvature, speed limit, lighting, weather, road signs, time of day, and more
- **Performance**: RMSE 0.0563 on validation set
- **Top Features**: Speed limit, lighting conditions, road curvature, weather

### Web Application
- **Framework**: Streamlit
- **Hosting**: Streamlit Community Cloud
- **Interactive Elements**: Real-time scoring, random scenario generation, educational insights

## 📊 Model Performance

The CatBoost model was trained on 517,754 road segments with the following results:
- **Validation RMSE**: 0.0563
- **Cross-validation RMSE**: 0.0561 ± 0.0005
- **Key Insights**: Speed limit and lighting conditions are the most important safety factors

## 🚀 Live Demo

Try the live app here:  
**[👉 https://road-safety-game.streamlit.app](https://road-safety-game.streamlit.app)**

## 📁 Project Structure

```
road-safety-game/
├── app.py                 # Streamlit application
├── requirements.txt       # Python dependencies
├── accident_risk_model.cbm # Trained CatBoost model
└── README.md             # This file
```

## 🏆 Challenge Information

This project is part of the **Kaggle & Stack Overflow Code Scientist Challenge**:
- **Kaggle Challenge**: Predict roadway accidents using machine learning
- **Stack Overflow Challenge**: Build an interactive web application to explore the data
- **Special Badge**: Complete both challenges to earn the "Code Scientist" badge

## 🔧 Installation & Local Development

If you want to run this locally:

```bash
# Clone the repository
git clone https://github.com/your-username/road-safety-game.git
cd road-safety-game

# Install dependencies
pip install -r requirements.txt

# Run the Streamlit app
streamlit run app.py
```

## 📈 Feature Importance

The model identified these as the most important safety factors:
1. **Speed Limit** (37.6%) - Higher speeds dramatically increase risk
2. **Lighting Conditions** (34.7%) - Poor visibility = higher risk
3. **Road Curvature** (14.4%) - Sharper curves = more dangerous
4. **Weather Conditions** (9.3%) - Rain, fog, snow increase accident risk

## 🤝 Contributing

Feel free to fork this project and submit pull requests for any improvements!

## 📄 License

This project is licensed under the MIT License - see the [LICENSE](LICENSE) file for details.

---

*Built with ❤️ for the Kaggle & Stack Overflow Code Scientist Challenge*
