import streamlit as st
import pandas as pd
import numpy as np
from catboost import CatBoostRegressor
import random

# Page configuration - WIDER LAYOUT
st.set_page_config(
    page_title="Road Safety Game",
    page_icon="🚗",
    layout="wide"  # Changed from "centered" to "wide"
)

# Load model
@st.cache_resource
def load_model():
    model = CatBoostRegressor()
    model.load_model('accident_risk_model.cbm')
    return model

# Initialize session state for game scoring
if 'score' not in st.session_state:
    st.session_state.score = 0
if 'total_questions' not in st.session_state:
    st.session_state.total_questions = 0
if 'current_roads' not in st.session_state:
    st.session_state.current_roads = None

# Load the model
try:
    model = load_model()
except Exception as e:
    st.error(f"Error loading model: {e}")
    st.stop()

# Define categorical features (same as in your training)
categorical_features = ['road_type', 'lighting', 'weather', 'road_signs_present', 
                       'public_road', 'time_of_day', 'holiday', 'school_season']

def generate_random_road():
    """Generate a random road scenario"""
    road_types = ['urban', 'rural', 'highway']
    lighting_conditions = ['daylight', 'dim', 'dark']
    weather_conditions = ['clear', 'rainy', 'foggy', 'snowy']
    times_of_day = ['morning', 'afternoon', 'evening', 'night']
    
    road = {
        'road_type': random.choice(road_types),
        'num_lanes': random.randint(1, 4),
        'curvature': round(random.uniform(0.0, 1.0), 2),
        'speed_limit': random.choice([25, 35, 45, 55, 65, 75]),
        'lighting': random.choice(lighting_conditions),
        'weather': random.choice(weather_conditions),
        'road_signs_present': random.choice([True, False]),
        'public_road': random.choice([True, False]),
        'time_of_day': random.choice(times_of_day),
        'holiday': random.choice([True, False]),
        'school_season': random.choice([True, False]),
        'num_reported_accidents': random.randint(0, 3)
    }
    
    return road

def predict_risk(road_data):
    """Predict accident risk for a road"""
    # Create DataFrame
    road_df = pd.DataFrame([road_data])
    
    # Convert boolean columns to strings (same as in your training)
    for col in ['road_signs_present', 'public_road', 'holiday', 'school_season']:
        road_df[col] = road_df[col].astype(str)
    
    # Make prediction
    risk = model.predict(road_df)[0]
    return max(0, min(1, risk))  # Ensure risk is between 0 and 1

def display_road_card(road, road_name):
    """Display road information in a card format"""
    with st.container():
        st.subheader(f"{road_name}")
        
        # Main safety factors - better layout
        col1, col2, col3 = st.columns(3)
        with col1:
            st.metric("Road Type", road['road_type'].title())
            st.metric("Lanes", road['num_lanes'])
        with col2:
            st.metric("Curvature", f"{road['curvature']:.2f}")
            st.metric("Speed Limit", f"{road['speed_limit']} mph")
        with col3:
            st.metric("Lighting", road['lighting'].title())
            st.metric("Weather", road['weather'].title())
        
        # Additional details in expander
        with st.expander("More details"):
            st.write(f"**Time:** {road['time_of_day'].title()}")
            st.write(f"**Road Signs:** {'✅ Yes' if road['road_signs_present'] else '❌ No'}")
            st.write(f"**Public Road:** {'✅ Yes' if road['public_road'] else '❌ No'}")
            st.write(f"**Holiday:** {'✅ Yes' if road['holiday'] else '❌ No'}")
            st.write(f"**School Season:** {'✅ Yes' if road['school_season'] else '❌ No'}")
            st.write(f"**Past Accidents:** {road['num_reported_accidents']}")

def show_risk_scores():
    """Display the actual risk scores - THIS WAS MISSING!"""
    risk1 = st.session_state.current_roads['risk1']
    risk2 = st.session_state.current_roads['risk2']
    st.info(f"**Actual Risk Scores:** Road 1: {risk1:.3f} | Road 2: {risk2:.3f}")
    
    # Show which factors mattered most
    st.write("**Key factors in this comparison:**")
    factors = []
    road1 = st.session_state.current_roads['road1']
    road2 = st.session_state.current_roads['road2']
    
    # Compare key features
    if road1['speed_limit'] != road2['speed_limit']:
        factors.append(f"Speed limit ({road1['speed_limit']} vs {road2['speed_limit']} mph)")
    if road1['lighting'] != road2['lighting']:
        factors.append(f"Lighting ({road1['lighting']} vs {road2['lighting']})")
    if road1['weather'] != road2['weather']:
        factors.append(f"Weather ({road1['weather']} vs {road2['weather']})")
    if road1['curvature'] != road2['curvature']:
        factors.append(f"Curvature ({road1['curvature']:.2f} vs {road2['curvature']:.2f})")
    if road1['road_type'] != road2['road_type']:
        factors.append(f"Road type ({road1['road_type']} vs {road2['road_type']})")
    
    if factors:
        for factor in factors[:4]:  # Show top 4 factors
            st.write(f"• {factor}")

# Main app - using wider layout
st.title("🚗 Pick the Safer Road")
st.markdown("### Test your intuition about road safety!")
st.markdown("Compare two road scenarios and choose which one you think has lower accident risk.")

# Game section
st.markdown("---")
st.header("🎮 The Game")

# Generate new roads button
if st.button("🎲 Generate New Road Scenarios", type="primary", use_container_width=True) or st.session_state.current_roads is None:
    road1 = generate_random_road()
    road2 = generate_random_road()
    
    risk1 = predict_risk(road1)
    risk2 = predict_risk(road2)
    
    st.session_state.current_roads = {
        'road1': road1, 'risk1': risk1,
        'road2': road2, 'risk2': risk2
    }
    st.rerun()

# Display current roads if they exist
if st.session_state.current_roads:
    road1 = st.session_state.current_roads['road1']
    road2 = st.session_state.current_roads['road2']
    
    # Display roads side by side - using wider columns
    col1, col2 = st.columns(2)
    
    with col1:
        display_road_card(road1, "Road 1 🛣️")
        
    with col2:
        display_road_card(road2, "Road 2 🛣️")
    
    st.markdown("---")
    st.subheader("🤔 Which road is safer?")
    st.write("*(Lower risk score = safer road)*")
    
    # Choice buttons - wider layout
    col1, col2, col3 = st.columns([1, 1, 1])
    
    with col1:
        if st.button("🚗 Choose Road 1", use_container_width=True, type="secondary"):
            st.session_state.total_questions += 1
            if st.session_state.current_roads['risk1'] < st.session_state.current_roads['risk2']:
                st.session_state.score += 1
                st.success("🎉 Correct! Road 1 is safer!")
            else:
                st.error("😞 Incorrect! Road 2 is safer.")
            
            show_risk_scores()
            st.session_state.current_roads = None
            st.rerun()
    
    with col2:
        if st.button("🤷‍♂️ Same Risk", use_container_width=True):
            st.session_state.total_questions += 1
            risk_diff = abs(st.session_state.current_roads['risk1'] - st.session_state.current_roads['risk2'])
            if risk_diff < 0.02:  # Very close risks
                st.session_state.score += 1
                st.success("🎉 Correct! Risks are very similar!")
            else:
                st.error("😞 Incorrect! One road is safer.")
            
            show_risk_scores()
            st.session_state.current_roads = None
            st.rerun()
    
    with col3:
        if st.button("🚗 Choose Road 2", use_container_width=True, type="secondary"):
            st.session_state.total_questions += 1
            if st.session_state.current_roads['risk2'] < st.session_state.current_roads['risk1']:
                st.session_state.score += 1
                st.success("🎉 Correct! Road 2 is safer!")
            else:
                st.error("😞 Incorrect! Road 1 is safer.")
            
            show_risk_scores()
            st.session_state.current_roads = None
            st.rerun()

# Score display
st.markdown("---")
st.header("📊 Your Score")
if st.session_state.total_questions > 0:
    accuracy = (st.session_state.score / st.session_state.total_questions) * 100
    col1, col2, col3 = st.columns(3)
    with col1:
        st.metric("Games Played", st.session_state.total_questions)
    with col2:
        st.metric("Correct Guesses", st.session_state.score)
    with col3:
        st.metric("Accuracy", f"{accuracy:.1f}%")
else:
    st.write("Play the game to see your score!")

# Reset score button
if st.session_state.total_questions > 0:
    if st.button("🔄 Reset Score", use_container_width=True):
        st.session_state.score = 0
        st.session_state.total_questions = 0
        st.rerun()

# Educational section
st.markdown("---")
st.header("💡 Did You Know?")
st.markdown("""
Based on the model's feature importance, here's what most affects road safety:

**Top Safety Factors:**
1. **Speed Limit** 🚀 - Higher speeds dramatically increase risk
2. **Lighting Conditions** 💡 - Poor visibility = higher risk  
3. **Road Curvature** 🌀 - Sharper curves = more dangerous
4. **Weather** 🌧️ - Rain, fog, snow increase accident risk

**Safety Tips:**
- Lower speeds save lives, especially in poor conditions
- Good lighting is crucial for night driving
- Adjust speed for road curvature and weather
- Always pay attention to road signs and conditions
""")

# Footer
st.markdown("---")
st.markdown(
    "Built using CatBoost | "
    "Data from [Kaggle Playground Series](https://www.kaggle.com/competitions/playground-series-s5e10) | "
    "Part of the Stack Overflow Code Scientist Challenge"
)
