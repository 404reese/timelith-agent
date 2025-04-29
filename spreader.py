import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import itertools
import random

st.set_page_config(page_title="Period even spread dis", layout="wide")

st.title("Period even spread dis")
st.write("Thinking : https://drive.google.com/file/d/1dgIcYXi2eigJurvcrTRKxoJ6PeL1JjzV/view?usp=drivesdk")
st.write("Code : https://github.com/404reese/timelith-agent/blob/main/spreader.py")

# Sidebar for inputs
with st.sidebar:
    st.header("Timetable Settings")
    num_days = st.slider("Number of Working Days", min_value=3, max_value=6, value=5)
    
    # Subject inputs
    st.subheader("Subject Configuration")
    num_subjects = st.number_input("Number of Subjects", min_value=1, max_value=10, value=5)
    
    subjects_data = []
    for i in range(num_subjects):
        st.markdown(f"**Subject {i+1}**")
        col1, col2 = st.columns(2)
        with col1:
            subject_name = st.text_input(f"Name #{i+1}", value=f"Subject {i+1}", key=f"subj_name_{i}")
        with col2:
            lectures_per_week = st.number_input(f"Lectures/Week #{i+1}", min_value=1, max_value=5, value=3, key=f"lectures_{i}")
        
        has_practical = st.checkbox(f"Has Practical #{i+1}", value=True, key=f"practical_{i}")
        
        subjects_data.append({
            "name": subject_name,
            "lectures": lectures_per_week,
            "has_practical": has_practical
        })

def create_timetable(subjects_data, num_days):
    """Generate a timetable based on subject requirements"""
    # Initialize empty timetable
    timetable = [[] for _ in range(num_days)]
    
    # First, distribute practicals evenly
    practical_subjects = [subj for subj in subjects_data if subj["has_practical"]]
    
    # Try to distribute practicals as evenly as possible
    practical_days = list(range(num_days))
    random.shuffle(practical_days)
    
    for i, subject in enumerate(practical_subjects):
        day = practical_days[i % num_days]
        timetable[day].append({
            'subject': subject["name"],
            'type': 'practical',
            'day': day
        })
    
    # Now distribute lectures
    for subject in subjects_data:
        # Get days that need fewer lectures (for balancing)
        day_loads = [len([e for e in day if e['type'] == 'lecture']) for day in timetable]
        
        # For each lecture of this subject
        for _ in range(subject["lectures"]):
            # Choose the day with the least lectures
            candidate_days = sorted(range(num_days), key=lambda d: day_loads[d])
            
            # Find days that don't already have this subject as a lecture
            available_days = [
                day for day in candidate_days 
                if not any(e['subject'] == subject["name"] and e['type'] == 'lecture' for e in timetable[day])
            ]
            
            # If we've used all days for this subject's lectures already, just use the least loaded day
            if not available_days:
                chosen_day = candidate_days[0]
            else:
                chosen_day = available_days[0]
            
            # Add the lecture
            timetable[chosen_day].append({
                'subject': subject["name"],
                'type': 'lecture',
                'day': chosen_day
            })
            
            # Update the load count for this day
            day_loads[chosen_day] += 1
    
    return timetable

def calculate_spread(timetable, event_type):
    """Calculate the distribution statistics for events in the timetable"""
    counts = [sum(1 for event in day if event['type'] == event_type) for day in timetable]
    if len(counts) == 0:
        return {'counts': counts, 'variance': 0, 'average': 0}
    average = sum(counts) / len(counts)
    variance = sum((x - average) ** 2 for x in counts) / len(counts)
    return {'counts': counts, 'variance': variance, 'average': average}

def optimize_timetable(subjects_data, num_days, iterations=100):
    """Generate multiple timetables and select the best one based on variance"""
    best_timetable = None
    best_score = float('inf')
    
    for _ in range(iterations):
        timetable = create_timetable(subjects_data, num_days)
        
        # Calculate spread stats
        lecture_stats = calculate_spread(timetable, 'lecture')
        practical_stats = calculate_spread(timetable, 'practical')
        
        # Combined score favoring even distribution (lower is better)
        total_score = lecture_stats['variance'] * 2 + practical_stats['variance']
        
        if total_score < best_score:
            best_score = total_score
            best_timetable = timetable
    
    return best_timetable

def display_timetable(timetable):
    """Create a DataFrame representation of the timetable"""
    days = ["Monday", "Tuesday", "Wednesday", "Thursday", "Friday", "Saturday"][:len(timetable)]
    
    # Create lecture and practical rows for each day
    data = []
    for day_idx, day_events in enumerate(timetable):
        lectures = [e['subject'] for e in day_events if e['type'] == 'lecture']
        practicals = [e['subject'] for e in day_events if e['type'] == 'practical']
        
        data.append({
            'Day': days[day_idx],
            'Lectures': ', '.join(lectures),
            'Practicals': ', '.join(practicals)
        })
    
    return pd.DataFrame(data)

def plot_distribution(timetable):
    """Plot the distribution of lectures and practicals"""
    days = ["Mon", "Tue", "Wed", "Thu", "Fri", "Sat"][:len(timetable)]
    
    lecture_counts = [sum(1 for e in day if e['type'] == 'lecture') for day in timetable]
    practical_counts = [sum(1 for e in day if e['type'] == 'practical') for day in timetable]
    
    fig, ax = plt.subplots(figsize=(10, 5))
    
    x = np.arange(len(days))
    width = 0.35
    
    ax.bar(x - width/2, lecture_counts, width, label='Lectures')
    ax.bar(x + width/2, practical_counts, width, label='Practicals')
    
    ax.set_xticks(x)
    ax.set_xticklabels(days)
    ax.set_ylabel('Count')
    ax.set_title('Distribution of Classes by Day')
    ax.legend()
    
    return fig

# Main content area
st.header("Generated Timetable")

if st.button("Generate Optimized Timetable"):
    if not subjects_data or all(not subj["name"].strip() for subj in subjects_data):
        st.error("Please add at least one subject")
    else:
        with st.spinner("Optimizing timetable..."):
            # Generate and optimize timetable
            timetable = optimize_timetable(subjects_data, num_days, iterations=200)
            
            # Display the timetable
            timetable_df = display_timetable(timetable)
            st.dataframe(timetable_df, use_container_width=True)
            
            # Create tabs for additional info
            tab1, tab2 = st.tabs(["Distribution Chart", "Statistics"])
            
            with tab1:
                st.pyplot(plot_distribution(timetable))
            
            with tab2:
                # Calculate spread statistics
                lecture_stats = calculate_spread(timetable, 'lecture')
                practical_stats = calculate_spread(timetable, 'practical')
                
                st.subheader("Distribution Analysis")
                col1, col2 = st.columns(2)
                
                with col1:
                    st.write("**Lecture Distribution**")
                    st.write(f"Average per day: {lecture_stats['average']:.2f}")
                    st.write(f"Variance: {lecture_stats['variance']:.2f} (Lower is better)")
                    st.write(f"Distribution: {lecture_stats['counts']}")
                
                with col2:
                    st.write("**Practical Distribution**")
                    st.write(f"Average per day: {practical_stats['average']:.2f}")
                    st.write(f"Variance: {practical_stats['variance']:.2f} (Lower is better)")
                    st.write(f"Distribution: {practical_stats['counts']}")
                
                # Subject allocation summary
                st.subheader("Subject Allocation")
                subject_summary = []
                
                for subject in set(e['subject'] for day in timetable for e in day):
                    lecture_days = [i for i, day in enumerate(timetable) if any(e['subject'] == subject and e['type'] == 'lecture' for e in day)]
                    practical_days = [i for i, day in enumerate(timetable) if any(e['subject'] == subject and e['type'] == 'practical' for e in day)]
                    
                    day_names = ["Monday", "Tuesday", "Wednesday", "Thursday", "Friday", "Saturday"][:num_days]
                    lecture_days_names = [day_names[d] for d in lecture_days]
                    practical_days_names = [day_names[d] for d in practical_days]
                    
                    subject_summary.append({
                        'Subject': subject,
                        'Lecture Days': ', '.join(lecture_days_names),
                        'Practical Days': ', '.join(practical_days_names) if practical_days_names else 'None',
                        'Total Lectures': len(lecture_days),
                        'Has Practical': len(practical_days) > 0
                    })
                
                st.dataframe(pd.DataFrame(subject_summary), use_container_width=True)

# Footer
st.markdown("---")