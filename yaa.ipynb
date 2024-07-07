import pandas as pd
import numpy as np
import ipywidgets as widgets
from IPython.display import display, clear_output

# Load the dataset
df = pd.read_excel('Datagalih.xlsx')

# Extract unique classes
unique_classes = df['stroke'].unique()

# Feature columns
feature_columns = df.columns[df.columns != 'stroke']

# Dropdown widgets for user input
dropdown_widgets = {}
for column in feature_columns:
    if column == 'gender':
        values = ['Male', 'Female']
    elif column == 'age':
        values = ['Dewasa', 'Lansia', 'Orang Tua', 'Remaja']
    elif column == 'ever_married':
        values = ['Yes', 'No']
    elif column == 'work_type':
        values = ['childern', 'Govt_job', 'Self-employed', 'Private']
    elif column == 'Residence_type':
        values = ['Urban', 'Rural']
    elif column == 'avg_glucose_level':
        values = ['0-100 mg/dl', '100-200 mg/dl', '>200 mg/dl']
    elif column == 'bmi':
        values = ['Obesitas', 'Normal', 'BB Berlebih', 'Kurus']
    elif column == 'smoking_status':
        values = ['smokes', 'never smoked', 'formerly smoked', 'Unknown']
    else:
        values = df[column].unique()

    dropdown_widgets[column] = widgets.Dropdown(options=values, description=f'{column}: ', style={'description_width': 'initial'})

# Submit button
submit_button = widgets.Button(description="Predict Stroke", button_style='success')

# Output widgets to display user input and prediction result
output_user_input = widgets.Output(layout={'border': '1px solid black', 'padding': '10px', 'margin': '10px 0'})
output_prediction_result = widgets.Output(layout={'border': '1px solid black', 'padding': '10px', 'margin': '10px 0'})

# Function to handle prediction when the submit button is clicked
def on_submit_button_clicked(b):
    # Get user input
    user_input = {}
    for column, dropdown in dropdown_widgets.items():
        user_input[column] = dropdown.value

    # Display user input
    with output_user_input:
        clear_output(wait=True)
        print("User Input:")
        for column, value in user_input.items():
            print(f"{column}: {value}")

    # Prepare user input for prediction
    user_input_df = pd.DataFrame([user_input])
    user_input_df = user_input_df[feature_columns]  # Rearrange columns if needed

    # Perform Naive Bayes prediction
    test_records = calc_posterior_prob(test_data=user_input_df,
                                       unique_classes=unique_classes,
                                       feature_columns=feature_columns,
                                       train_data=X_train,
                                       train_record=train_record)

    predictions = get_max_class_prediction(test_records=test_records,
                                           unique_classes=unique_classes)

    # Display prediction result
    with output_prediction_result:
        clear_output(wait=True)
        print("Prediction Result:")
        print(f"Stroke Prediction: {predictions[0]}")  # Assuming binary classification, adjust accordingly

# Set the function to be called when the button is clicked
submit_button.on_click(on_submit_button_clicked)

# Display widgets
user_input_widgets = [dropdown_widgets[column] for column in feature_columns]
user_input_widgets.append(submit_button)

# Center align user input and prediction result
output_user_input.layout.align_items = 'center'
output_prediction_result.layout.align_items = 'center'

display(widgets.VBox(user_input_widgets))
display(output_user_input)
display(output_prediction_result)