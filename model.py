import streamlit as st
import pickle
from PIL import Image
from streamlit_option_menu import option_menu
import pandas as pd
import altair as alt

# Load the dataset
dataset_name = 'clean-data.csv'
df = pd.read_csv(dataset_name)

#sidebar menu
with st.sidebar:
    selected = option_menu(
        menu_title="Main menu",
        options=["Model results","Input","Analysis"],
        menu_icon="cast",
        default_index=0,
    )
st.title("Financial Status Analysis using ML")

if selected =="Model results":
    st.title(f"These are the results of our **Machine Learning** model")
# Load the saved results
    with open('random_forest_cls.pkl', 'rb') as file:
        dt_model = pickle.load(file)

    with open('grid_search_random_forest.pkl', 'rb') as file:
        grid_search_results = pickle.load(file)

    with open('scaler.pkl', 'rb') as file:
        scaler = pickle.load(file)

    with open('cm_random_forest.pkl', 'rb') as file:
        confusion_mat = pickle.load(file)

    with open('classification_report.pkl', 'rb') as file:
        class_report = pickle.load(file)

# Streamlit app

# Display the loaded results
    st.markdown("<style>h1 { font-size: 30px; }</style>", unsafe_allow_html=True)
    st.markdown("<h1>Decision Tree Model:</h1>", unsafe_allow_html=True)


    st.markdown("A Decision Tree is a supervised machine learning algorithm used for both classification and regression tasks. "
            "The model is built by recursively splitting the dataset based on features to create a tree-like structure. "
            "At each node of the tree, the algorithm selects the feature that best separates the data, "
            "resulting in a hierarchy of decision nodes until a leaf node is reached.")

    st.markdown("### Key Concepts:")
    st.markdown("- **Root Node:** The top node of the tree where the first split occurs.")
    st.markdown("- **Decision Nodes (Interior Nodes):** Nodes where the dataset is split based on a feature.")
    st.markdown("- **Leaf Nodes:** Terminal nodes where the final prediction is made.")
    st.markdown("- **Splitting Criteria:** The method used to decide how to split the data at each node (e.g., Gini impurity, information gain).")

# Display Decision Tree 

    st.markdown("<style>h1 { font-size: 30px; }</style>", unsafe_allow_html=True)
    st.markdown("<h1>Grid Search Results:</h1>", unsafe_allow_html=True)

    st.markdown("Grid Search is a technique used for hyperparameter tuning to find the best set of hyperparameters for a model. "
            "It performs an exhaustive search over a specified parameter grid and cross-validates the results to find the optimal combination.")


    st.markdown("<style>h1 { font-size: 30px; }</style>", unsafe_allow_html=True)
    st.markdown("<h1>Scaler:</h1>", unsafe_allow_html=True)



    st.markdown("Standard Scaler is a preprocessing technique used to standardize features by removing the mean and scaling to unit variance. "
            "It helps in bringing all features to the same scale, preventing some features from dominating due to their larger magnitude.")


    st.write(scaler)

    st.subheader("Confusion Matrix:")

    st.markdown("A Confusion Matrix is a table that is often used to describe the performance of a classification model on a set of test data. "
            "It shows the counts of true positive, true negative, false positive, and false negative predictions. "
            "From the confusion matrix, various performance metrics like accuracy, precision, recall, and F1 score can be derived.")

#st.image(confusion_matrix(confusion_mat).plot().figure_, use_column_width=True)
    image = Image.open('cm_random_forest.png')
    st.image(image, use_column_width=True)
    st.subheader("Classification Report:")

    st.markdown("The Classification Report is a summary of the precision, recall, F1-score, and support for each class in the classification problem. "
            "It provides a comprehensive view of the model's performance, especially in a multi-class setting.")

    st.markdown("### Key Metrics:")
    st.markdown("- **Precision:** The ratio of true positive predictions to the total predicted positives. Precision is a measure of the accuracy of the positive predictions.")
    st.markdown("- **Recall (Sensitivity or True Positive Rate):** The ratio of true positive predictions to the total actual positives. Recall measures the ability of the model to capture all positive instances.")
    st.markdown("- **F1-score:** The harmonic mean of precision and recall. It provides a balance between precision and recall.")
    st.markdown("- **Support:** The number of actual occurrences of the class in the specified dataset.")


    image2 = Image.open('classification_report.PNG')
    st.image(image2, use_column_width=True)
    classification_report_text = """
    Here's the interpretation for each class:

**Class 0:**
- Precision: 0.77 (77% of predicted class 0 instances were correct)
- Recall: 0.76 (76% of actual class 0 instances were captured)
- F1-score: 0.77 (balanced measure of precision and recall)
- Support: 4736 instances of class 0 in the dataset

**Class 1:**
- Precision: 0.78 (78% of predicted class 1 instances were correct)
- Recall: 0.82 (82% of actual class 1 instances were captured)
- F1-score: 0.80 (balanced measure of precision and recall)
- Support: 9002 instances of class 1 in the dataset

**Class 2:**
- Precision: 0.70 (70% of predicted class 2 instances were correct)
- Recall: 0.62 (62% of actual class 2 instances were captured)
- F1-score: 0.66 (balanced measure of precision and recall)
- Support: 2794 instances of class 2 in the dataset

**Macro Avg (Macro Average):**
- Average precision: 0.75
- Average recall: 0.73
- Average F1-score: 0.74

**Weighted Avg (Weighted Average):**
- Weighted precision: 0.77
- Weighted recall: 0.77
- Weighted F1-score: 0.77

**Accuracy:**
- Overall accuracy of the model across all classes: 0.77 (77%)
"""

# Display classification report in Streamlit app
    st.markdown(classification_report_text)




if selected =="Input":
    #st.subheader(f"Fill me!")
    
    # Function to calculate credit score and financial eligibility
    def calculate_credit_score_and_eligibility(user_input):
    # Extract user input
        age = user_input['age_default']
        annual_income = user_input['annual_income_default']
        accounts = user_input['accounts_default']
        credit_cards = user_input['credit_cards_default']
        delayed_payments = user_input['delayed_payments_default']
        credit_card_ratio = user_input['credit_card_ratio_default']
        emi_monthly = user_input['emi_monthly_default']
        credit_history = user_input['credit_history_default']
        loans = user_input['loans_default']
        missed_payment = user_input['missed_payment_default']
        minimum_payment = user_input['minimum_payment_default']

    # Calculate credit score (adjust scoring logic as needed)
        credit_score = (
        age * 10 +
        annual_income * 0.1 +
        accounts * 5 +
        credit_cards * 5 +
        delayed_payments * -10 +
        credit_card_ratio * -20 +
        emi_monthly * -5 +
        credit_history * 5 +
        loans * -10 +
        missed_payment * -20 +
        minimum_payment * -5
    )

    # Ensure the credit score is within the range [300, 850]
        credit_score = max(300, min(850, credit_score))

    # Determine financial eligibility based on credit score and provide recommendations
        if credit_score >= 720:
            st.balloons()
            eligibility = "Eligible"
            recommendations = "Congratulations! You have a high credit score. You are eligible for favorable financial terms."
        elif credit_score >= 650:
            eligibility = "Eligible with conditions"
            recommendations = "You are eligible, but some conditions may apply. Consider improving specific aspects of your financial profile for better terms."
        else:
            eligibility = "Not Eligible"
            recommendations = "Your credit score is below the threshold. Consider improving your financial profile to become eligible for better terms."

        return credit_score, eligibility, recommendations
    st.subheader("Fill in the details for credit score calculation:")  
    age_default = st.number_input("Age", min_value=18, max_value=100, value=25)
    annual_income_default = st.number_input("Annual Income", min_value=0, value=50000)
    accounts_default = st.number_input("Number of Accounts", min_value=0, value=2)
    credit_cards_default = st.number_input("Number of Credit Cards", min_value=0, value=1)
    delayed_payments_default = st.number_input("Delayed Payments (months)", min_value=0, value=0)
    credit_card_ratio_default = st.number_input("Credit Card Ratio", min_value=0.0, value=0.3, max_value=1.0, step=0.01)
    emi_monthly_default = st.number_input("Monthly EMI", min_value=0, value=1000)
    credit_history_default = st.number_input("Credit History", min_value=0, max_value=10, value=5)
    loans_default = st.number_input("Number of Loans", min_value=0, value=1)
    missed_payment_default = st.number_input("Missed Payments", min_value=0, value=0)
    minimum_payment_default = st.number_input("Minimum Payments", min_value=0, value=500)

    user_input = {
            'age_default': age_default,
            'annual_income_default': annual_income_default,
            'accounts_default': accounts_default,
            'credit_cards_default': credit_cards_default,
            'delayed_payments_default': delayed_payments_default,
            'credit_card_ratio_default': credit_card_ratio_default,
            'emi_monthly_default': emi_monthly_default,
            'credit_history_default': credit_history_default,
            'loans_default': loans_default,
            'missed_payment_default': missed_payment_default,
            'minimum_payment_default': minimum_payment_default,
        }

    if st.button("Calculate Credit Score"):
        credit_score, eligibility, recommendations = calculate_credit_score_and_eligibility(user_input)
        st.subheader(f"Calculated Credit Score: {credit_score:.2f}")
        st.subheader(f"Financial Eligibility: {eligibility}")
        st.subheader("Recommendations:")
        st.write(recommendations)
# Load the model and scaler









if selected == "Analysis":
    st.subheader("Financial Status Analysis")

    # Function to perform financial status analysis with visualizations
    def perform_analysis_with_visualizations(user_input, df):
        # Extract user input
        example_id = user_input['ID']  # Fix here to match the key in your user_input dictionary

        # Fetch the corresponding row from the dataset
        user_data = df[df['Customer_ID'] == example_id]  # Fix here to match the column name in your dataset

        if user_data.empty:
            return "No data found for the provided ID."

        # Display relevant visualizations and analysis for each column
        st.subheader("User Data:")
        st.write(user_data)

        st.subheader("Column-wise Analysis:")
        for column in user_data.columns:
            st.write(f"### {column}")

            # Check if all 8 months have the same value for the current column
            if user_data[column].nunique() == 1:
                st.write("All 8 months have the same value. Original instance value:", user_data[column].iloc[0])
            else:
                # Visualize data trends or patterns (you can customize the visualization based on the column type)
                if user_data[column].dtype in ['int64', 'float64']:  # Numeric columns
                    # Create an interactive line chart using Altair
                    chart = alt.Chart(user_data).mark_line().encode(
                        x='Month:T',  # Explicitly set the data type to temporal
                        y=column,
                        tooltip=[column]
                    ).properties(
                        width=600,
                        height=300
                    )
                    st.altair_chart(chart)
                elif user_data[column].dtype == 'object':  # Categorical columns
                    st.bar_chart(user_data[column].value_counts())
                # Add more conditions as needed for different column types

        return "Analysis completed."

    # User input for financial analysis
    st.subheader("Provide input for financial status analysis:")
    example_id_analysis = st.text_input("ID or Customer_ID", "12345")  # Replace "12345" with an actual ID from your dataset

    analysis_input = {
        'ID': example_id_analysis,  # Fix here to match the key in your user_input dictionary
        # Add more fields as needed
    }

    # Perform analysis when the user clicks the button
    if st.button("Perform Analysis"):
        analysis_result = perform_analysis_with_visualizations(analysis_input, df)

        # Display the analysis result
        st.subheader("Analysis Result:")
        st.write(analysis_result)
































