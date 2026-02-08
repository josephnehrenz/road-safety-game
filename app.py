import streamlit as st
import pandas as pd
import numpy as np
from catboost import CatBoostRegressor
import random
import plotly.graph_objects as go
import plotly.express as px

# Page configuration - WIDER LAYOUT
st.set_page_config(
    page_title="Road Safety Game",
    page_icon="🚗",
    layout="wide"
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
if 'game_history' not in st.session_state:
    st.session_state.game_history = []
if 'game_complete' not in st.session_state:
    st.session_state.game_complete = False
if 'high_scores' not in st.session_state:
    st.session_state.high_scores = []  # Store multiple high scores
if 'mistake_analysis' not in st.session_state:
    st.session_state.mistake_analysis = {
        'speed_limit': 0,
        'lighting': 0,
        'curvature': 0,
        'weather': 0,
        'road_type': 0,
        'time_of_day': 0,
        'num_lanes': 0
    }

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
    """Display road information in a card format - CLEANER LAYOUT"""
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
        
        st.markdown("---")
        st.write("**Additional Details:**")
        
        # Additional Details
        detail_col1, detail_col2 = st.columns(2)
        
        with detail_col1:
            st.write(f"**⏰ Time of Day:** {road['time_of_day'].title()}")
            st.write(f"**🛑 Road Signs:** {'✅ Present' if road['road_signs_present'] else '❌ Not Present'}")
            st.write(f"**🏛️ Public Road:** {'✅ Yes' if road['public_road'] else '❌ No'}")
            st.write(f"**📊 Past Accidents:** {road['num_reported_accidents']}")
            
        with detail_col2:
            st.write(f"**🎄 Holiday:** {'✅ Yes' if road['holiday'] else '❌ No'}")
            st.write(f"**🏫 School Season:** {'✅ Yes' if road['school_season'] else '❌ No'}")

def show_risk_scores():
    """Display the actual risk scores"""
    risk1 = st.session_state.current_roads['risk1']
    risk2 = st.session_state.current_roads['risk2']
    st.info(f"**Actual Risk Scores:** Road 1: {risk1:.3f} | Road 2: {risk2:.3f}")
    
    # Show which factors mattered most
    st.write("**Key factors in this comparison:**")
    factors = []
    road1 = st.session_state.current_roads['road1']
    road2 = st.session_state.current_roads['road2']
    
    # Compare key features (ordered by importance)
    if road1['speed_limit'] != road2['speed_limit']:
        factors.append(f"🚀 Speed limit ({road1['speed_limit']} vs {road2['speed_limit']} mph)")
    if road1['lighting'] != road2['lighting']:
        factors.append(f"💡 Lighting ({road1['lighting']} vs {road2['lighting']})")
    if road1['weather'] != road2['weather']:
        factors.append(f"🌧️ Weather ({road1['weather']} vs {road2['weather']})")
    if road1['curvature'] != road2['curvature']:
        factors.append(f"🌀 Curvature ({road1['curvature']:.2f} vs {road2['curvature']:.2f})")
    if road1['road_type'] != road2['road_type']:
        factors.append(f"🛣️ Road type ({road1['road_type']} vs {road2['road_type']})")
    if road1['time_of_day'] != road2['time_of_day']:
        factors.append(f"⏰ Time ({road1['time_of_day']} vs {road2['time_of_day']})")
    if road1['num_lanes'] != road2['num_lanes']:
        factors.append(f"↔️ Lanes ({road1['num_lanes']} vs {road2['num_lanes']})")
    
    if factors:
        for factor in factors:
            st.write(f"• {factor}")

def create_progress_chart():
    """Create a progress chart showing accuracy over time - FIXED VERSION"""
    if len(st.session_state.game_history) == 0:
        return None
    
    # FIXED: Calculate cumulative accuracy correctly
    questions = list(range(1, len(st.session_state.game_history) + 1))
    cumulative_correct = np.cumsum([1 if correct else 0 for correct in st.session_state.game_history])
    cumulative_accuracy = [(cumulative_correct[i] / (i + 1)) * 100 for i in range(len(cumulative_correct))]
    
    fig = go.Figure()
    
    # Add accuracy line
    fig.add_trace(go.Scatter(
        x=questions,
        y=cumulative_accuracy,
        mode='lines+markers',
        name='Accuracy',
        line=dict(color='#1f77b4', width=3),
        marker=dict(size=8)
    ))
    
    # Add perfect score reference line
    fig.add_hline(y=100, line_dash="dash", line_color="green", opacity=0.3)
    # Add 50% reference line
    fig.add_hline(y=50, line_dash="dash", line_color="orange", opacity=0.3)
    
    fig.update_layout(
        title="Your Accuracy Progress",
        xaxis_title="Question Number",
        yaxis_title="Accuracy (%)",
        height=300,
        showlegend=False,
        yaxis=dict(range=[0, 100])  # Always show 0-100% scale
    )
    
    return fig

def create_score_gauge():
    """Create a gauge chart for final score"""
    final_score = st.session_state.score
    total_questions = st.session_state.total_questions
    accuracy = (final_score / total_questions) * 100 if total_questions > 0 else 0
    
    # Determine color based on performance
    if accuracy >= 80:
        gauge_color = "green"
    elif accuracy >= 60:
        gauge_color = "orange"
    else:
        gauge_color = "red"
    
    fig = go.Figure(go.Indicator(
        mode = "gauge+number+delta",
        value = accuracy,
        domain = {'x': [0, 1], 'y': [0, 1]},
        title = {'text': f"Final Score: {final_score}/{total_questions}"},
        delta = {'reference': 50},
        gauge = {
            'axis': {'range': [None, 100]},
            'bar': {'color': gauge_color},
            'steps': [
                {'range': [0, 50], 'color': "lightgray"},
                {'range': [50, 80], 'color': "gray"},
                {'range': [80, 100], 'color': "lightgray"}],
            'threshold': {
                'line': {'color': "red", 'width': 4},
                'thickness': 0.75,
                'value': 90}}
    ))
    
    fig.update_layout(height=300)
    return fig

def update_mistake_analysis(road1, road2, user_choice_correct):
    """Analyze which factors might have caused mistakes"""
    if user_choice_correct:
        return  # Don't analyze correct choices
    
    # Check which important features were different between the roads
    if road1['speed_limit'] != road2['speed_limit']:
        st.session_state.mistake_analysis['speed_limit'] += 1
    if road1['lighting'] != road2['lighting']:
        st.session_state.mistake_analysis['lighting'] += 1
    if road1['curvature'] != road2['curvature']:
        st.session_state.mistake_analysis['curvature'] += 1
    if road1['weather'] != road2['weather']:
        st.session_state.mistake_analysis['weather'] += 1
    if road1['road_type'] != road2['road_type']:
        st.session_state.mistake_analysis['road_type'] += 1
    if road1['time_of_day'] != road2['time_of_day']:
        st.session_state.mistake_analysis['time_of_day'] += 1
    if road1['num_lanes'] != road2['num_lanes']:
        st.session_state.mistake_analysis['num_lanes'] += 1

def create_mistake_analysis_chart():
    """Create a chart showing which factors caused the most mistakes"""
    if sum(st.session_state.mistake_analysis.values()) == 0:
        return None
    
    # Prepare data for plotting
    factors = list(st.session_state.mistake_analysis.keys())
    counts = list(st.session_state.mistake_analysis.values())
    
    # Create human-readable labels with emojis
    factor_labels = {
        'speed_limit': '🚀 Speed Limit',
        'lighting': '💡 Lighting',
        'curvature': '🌀 Curvature',
        'weather': '🌧️ Weather',
        'road_type': '🛣️ Road Type',
        'time_of_day': '⏰ Time of Day',
        'num_lanes': '↔️ Number of Lanes'
    }
    
    readable_factors = [factor_labels[factor] for factor in factors]
    
    fig = px.bar(
        x=counts,
        y=readable_factors,
        orientation='h',
        title="📊 Factors That Challenged You Most",
        labels={'x': 'Number of Mistakes', 'y': 'Road Features'},
        color=counts,
        color_continuous_scale='Reds'
    )
    
    fig.update_layout(
        height=300,
        showlegend=False,
        xaxis=dict(range=[0, max(counts) + 1])
    )
    
    # Add count labels on bars
    fig.update_traces(
        texttemplate='%{x}',
        textposition='outside'
    )
    
    return fig

def get_improvement_tips():
    """Generate personalized improvement tips based on mistake analysis"""
    if sum(st.session_state.mistake_analysis.values()) == 0:
        return ["Great job! You didn't make any consistent mistakes."]
    
    # Find the top 2-3 factors with most mistakes
    sorted_factors = sorted(st.session_state.mistake_analysis.items(), 
                          key=lambda x: x[1], reverse=True)
    
    tips = []
    tip_messages = {
        'speed_limit': "**🚀 Speed Limits**: Higher speeds dramatically increase accident risk.",
        'lighting': "**💡 Lighting**: Poor visibility is a major risk factor.",
        'curvature': "**🌀 Curvature**: Sharper curves are much more dangerous.",
        'weather': "**🌧️ Weather**: Rain/fog/snow can triple accident risk.",
        'road_type': "**🛣️ Road Types**: Highways > rural > urban for safety.",
        'time_of_day': "**⏰ Time**: Nighttime driving is riskier than daytime.",
        'num_lanes': "**↔️ Lanes**: More lanes can mean more complex driving."
    }
    
    for factor, count in sorted_factors[:3]:  # Top 3 factors
        if count > 0:
            tips.append(tip_messages.get(factor, ""))
    
    return tips

def update_high_scores():
    """Update high scores with current game result"""
    final_score = st.session_state.score
    total_questions = st.session_state.total_questions
    accuracy = (final_score / total_questions) * 100
    
    # Add current score to high scores
    st.session_state.high_scores.append({
        'score': final_score,
        'total': total_questions,
        'accuracy': accuracy,
        'timestamp': pd.Timestamp.now().strftime("%Y-%m-%d %H:%M")
    })
    
    # Sort by accuracy (descending) and keep top 5
    st.session_state.high_scores.sort(key=lambda x: x['accuracy'], reverse=True)
    st.session_state.high_scores = st.session_state.high_scores[:5]

def create_high_scores_chart():
    """Create a bar chart of high scores"""
    if not st.session_state.high_scores:
        return None
    
    # Prepare data for plotting
    scores_df = pd.DataFrame(st.session_state.high_scores)
    scores_df['label'] = [f"Game {i+1}" for i in range(len(scores_df))]
    
    fig = px.bar(
        scores_df,
        x='label',
        y='accuracy',
        title="🏆 High Scores History",
        labels={'accuracy': 'Accuracy (%)', 'label': 'Game'},
        color='accuracy',
        color_continuous_scale='Viridis'
    )
    
    fig.update_layout(
        height=300,
        showlegend=False,
        yaxis=dict(range=[0, 100])
    )
    
    # Add score labels on bars
    fig.update_traces(
        texttemplate='%{y:.1f}%',
        textposition='outside'
    )
    
    return fig

def check_game_complete():
    """Check if the 10-question game is complete"""
    if st.session_state.total_questions >= 10:
        st.session_state.game_complete = True
        update_high_scores()

# Main app - using wider layout
st.title("🚗 Pick the Safer Road")
st.markdown("### Test your intuition about road safety!")
st.markdown("Compare two road scenarios and choose which one you think has lower accident risk.")

# Game section
st.markdown("---")
st.header("🎮 The Game")

# Show game progress
if st.session_state.total_questions > 0 and not st.session_state.game_complete:
    progress = st.session_state.total_questions / 10
    st.progress(progress, text=f"Progress: {st.session_state.total_questions}/10 questions")

# Game complete message
if st.session_state.game_complete:
    st.balloons()
    st.success("🎉 Congratulations! You've completed the 10-question challenge!")
    
    # Show improvement analysis at the TOP
    mistake_chart = create_mistake_analysis_chart()
    improvement_tips = get_improvement_tips()
    
    if mistake_chart:
        st.subheader("📈 Your Learning Analysis")
        col1, col2 = st.columns(2)
        with col1:
            st.plotly_chart(mistake_chart, use_container_width=True, key="mistake_chart")
        with col2:
            st.subheader("💡 Tips for Improvement")
            for tip in improvement_tips:
                st.write(tip)
    
    # Show final results
    st.subheader("🎯 Final Score")
    col1, col2 = st.columns(2)
    with col1:
        # Use unique key to avoid duplicate element error
        st.plotly_chart(create_score_gauge(), use_container_width=True, key="final_gauge")
    with col2:
        st.plotly_chart(create_progress_chart(), use_container_width=True, key="final_progress")
    
    # Show high scores
    st.subheader("🏆 High Scores")
    if st.session_state.high_scores:
        col1, col2 = st.columns(2)
        with col1:
            # Display high scores table
            scores_df = pd.DataFrame(st.session_state.high_scores)
            st.dataframe(
                scores_df[['score', 'total', 'accuracy', 'timestamp']].rename(
                    columns={'score': 'Correct', 'total': 'Total', 'accuracy': 'Accuracy%', 'timestamp': 'Date'}
                ),
                use_container_width=True
            )
        with col2:
            st.plotly_chart(create_high_scores_chart(), use_container_width=True, key="high_scores_chart")
    else:
        st.info("Play more games to build your high scores history!")
    
    # Reset button
    if st.button("🔄 Play Again", type="primary", use_container_width=True):
        st.session_state.score = 0
        st.session_state.total_questions = 0
        st.session_state.game_history = []
        st.session_state.game_complete = False
        st.session_state.current_roads = None
        st.session_state.mistake_analysis = {key: 0 for key in st.session_state.mistake_analysis}
        st.rerun()

# Generate new roads button (only if game not complete) - CHANGED TO BLUE
if not st.session_state.game_complete:
    if st.button("🎲 Generate New Road Scenarios", type="secondary", use_container_width=True, 
                help="Click to get two new random road scenarios to compare") or st.session_state.current_roads is None:
        road1 = generate_random_road()
        road2 = generate_random_road()
        
        risk1 = predict_risk(road1)
        risk2 = predict_risk(road2)
        
        st.session_state.current_roads = {
            'road1': road1, 'risk1': risk1,
            'road2': road2, 'risk2': risk2
        }
        st.rerun()

# Display current roads if they exist and game not complete
if st.session_state.current_roads and not st.session_state.game_complete:
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
    
    def handle_choice(correct_condition, chosen_road):
        st.session_state.total_questions += 1
        is_correct = correct_condition
        
        if is_correct:
            st.session_state.score += 1
            st.success(f"🎉 Correct! {chosen_road} is safer!")
        else:
            st.error(f"😞 Incorrect! The other road is safer.")
            # Analyze mistake
            update_mistake_analysis(
                st.session_state.current_roads['road1'],
                st.session_state.current_roads['road2'],
                is_correct
            )
        
        # Record game history
        st.session_state.game_history.append(is_correct)
        show_risk_scores()
        check_game_complete()
        st.session_state.current_roads = None
    
    with col1:
        if st.button("🚗 Choose Road 1", use_container_width=True, type="secondary"):
            handle_choice(
                st.session_state.current_roads['risk1'] < st.session_state.current_roads['risk2'],
                "Road 1"
            )
            st.rerun()
    
    with col2:
        if st.button("🤷‍♂️ Same Risk", use_container_width=True):
            risk_diff = abs(st.session_state.current_roads['risk1'] - st.session_state.current_roads['risk2'])
            handle_choice(risk_diff < 0.02, "Both roads have similar risk")
            st.rerun()
    
    with col3:
        if st.button("🚗 Choose Road 2", use_container_width=True, type="secondary"):
            handle_choice(
                st.session_state.current_roads['risk2'] < st.session_state.current_roads['risk1'],
                "Road 2"
            )
            st.rerun()

# Score display
st.markdown("---")
st.header("📊 Your Score")

if st.session_state.total_questions > 0:
    accuracy = (st.session_state.score / st.session_state.total_questions) * 100
    
    # Current stats
    col1, col2, col3, col4 = st.columns(4)
    with col1:
        st.metric("Games Played", st.session_state.total_questions)
    with col2:
        st.metric("Correct Guesses", st.session_state.score)
    with col3:
        st.metric("Accuracy", f"{accuracy:.1f}%")
    with col4:
        remaining = 10 - st.session_state.total_questions
        st.metric("Questions Left", f"{remaining}" if not st.session_state.game_complete else "Complete!")
    
    # Progress chart (only show if we have history and game not complete)
    if len(st.session_state.game_history) > 1 and not st.session_state.game_complete:
        # Use unique key for each chart instance
        st.plotly_chart(create_progress_chart(), use_container_width=True, key="progress_chart")
    
    # Show high scores if we have any (even during game)
    if st.session_state.high_scores and not st.session_state.game_complete:
        st.subheader("🏆 Your Best Scores")
        best_score = max([score['accuracy'] for score in st.session_state.high_scores])
        st.metric("Personal Best", f"{best_score:.1f}%")
else:
    st.write("Play the game to see your score!")

# Reset score button (only show if game in progress)
if st.session_state.total_questions > 0 and not st.session_state.game_complete:
    if st.button("🔄 Reset Game", use_container_width=True):
        st.session_state.score = 0
        st.session_state.total_questions = 0
        st.session_state.game_history = []
        st.session_state.game_complete = False
        st.session_state.current_roads = None
        st.session_state.mistake_analysis = {key: 0 for key in st.session_state.mistake_analysis}
        st.rerun()

# Educational section - TWO COLUMN LAYOUT (BALANCED 6 & 6)
st.markdown("---")
st.header("💡 Did You Know?")

col1, col2 = st.columns(2)

with col1:
    st.subheader("🏆 Top Safety Factors")
    st.markdown("""
    1. **Speed Limit** 🚀 - Higher speeds dramatically increase risk
    2. **Lighting Conditions** 💡 - Poor visibility = higher risk  
    3. **Road Curvature** 🌀 - Sharper curves = more dangerous
    4. **Weather** 🌧️ - Rain, fog, snow increase accident risk
    5. **Road Type** 🛣️ - Highway vs urban vs rural differences
    6. **Time of Day** ⏰ - Nighttime driving increases risk
    """)

with col2:
    st.subheader("💡 Safety Tips")
    st.markdown("""
    - **Adjust speed** for road conditions and visibility
    - **Use headlights** in poor lighting and bad weather
    - **Slow down** on curved roads and unfamiliar routes
    - **Stay alert** for road signs and changing conditions
    - **Plan ahead** for holiday travel and school zones
    - **Maintain focus** - avoid distractions while driving
    """)

# Footer
st.markdown("---")
st.markdown(
    "Built using CatBoost | "
    "Data from [Kaggle Playground Series](https://www.kaggle.com/competitions/playground-series-s5e10) | "
    "Part of the Stack Overflow Code Scientist Challenge"
)
