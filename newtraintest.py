
import pandas as pd
import numpy as np
import pickle
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from sklearn.compose import ColumnTransformer
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
from tensorflow.keras.models import Sequential, Model
from tensorflow.keras.layers import LSTM, SimpleRNN, GRU, Dense, Input, Dropout, Flatten, MultiHeadAttention, LayerNormalization
from tensorflow.keras.callbacks import EarlyStopping, ReduceLROnPlateau
from tensorflow.keras.regularizers import l2
import os
base_dir = "e:/MCU/Final Project/test061224"
charts_dir = os.path.join(base_dir, "charts")
os.makedirs(charts_dir, exist_ok=True)

model_save_dir = base_dir  # Directory to save models
os.makedirs(model_save_dir, exist_ok=True)


# Load Dataset
data = pd.read_csv('e:/MCU/Final Project/test061224/cleaned_vehicles_updated_year_new.csv', low_memory=False)
data = data.dropna(subset=['displ', 'cylinders', 'city08', 'highway08', 'comb08', 'co2'])

# Preprocess Data
numerical_features = ['displ', 'cylinders', 'city08', 'highway08', 'comb08']
categorical_features = ['make', 'model', 'VClass', 'trany', 'fuelType']

preprocessor = ColumnTransformer(
    transformers=[
        ('num', StandardScaler(), numerical_features),
        ('cat', OneHotEncoder(handle_unknown='ignore'), categorical_features)
    ]
)

X = data[numerical_features + categorical_features]
y = data['comb08']

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

X_train_preprocessed = preprocessor.fit_transform(X_train)
X_test_preprocessed = preprocessor.transform(X_test)

# Save preprocessor for later use in Streamlit
with open('preprocessor.pkl', 'wb') as f:
    pickle.dump(preprocessor, f)

y_train = np.array(y_train).reshape(-1, 1)
y_test = np.array(y_test).reshape(-1, 1)

X_train_lstm = X_train_preprocessed.toarray().reshape(X_train_preprocessed.shape[0], 1, X_train_preprocessed.shape[1])
X_test_lstm = X_test_preprocessed.toarray().reshape(X_test_preprocessed.shape[0], 1, X_test_preprocessed.shape[1])

# Metrics Calculation Function
def calculate_metrics(y_true, y_pred):
    mae = mean_absolute_error(y_true, y_pred)
    rmse = np.sqrt(mean_squared_error(y_true, y_pred))
    r2 = r2_score(y_true, y_pred)
    return mae, rmse, r2

# Additional Metrics Calculation Function
def calculate_additional_metrics(y_true, y_pred):
    """
    Calculate additional metrics: MSE, MAPE (%), SMAPE (%).
    :param y_true: Actual values
    :param y_pred: Predicted values
    :return: Tuple of (MSE, MAPE, SMAPE)
    """
    mse = mean_squared_error(y_true, y_pred)  # Mean Squared Error
    mape = np.mean(np.abs((y_true - y_pred) / y_true)) * 100  # Mean Absolute Percentage Error
    smape = 100 * np.mean(2 * np.abs(y_true - y_pred) / (np.abs(y_true) + np.abs(y_pred)))  # Symmetric MAPE
    return mse, mape, smape

# Plot Training History as Line Charts
def plot_training_history(history, model_name):
    plt.figure(figsize=(8, 5))
    plt.plot(history.history['loss'], label='Training Loss')
    if 'val_loss' in history.history:
        plt.plot(history.history['val_loss'], label='Validation Loss')
    plt.title(f'{model_name} Training History')
    plt.xlabel('Epochs')
    plt.ylabel('Loss')
    plt.legend()
    plt.grid()
    plt.show()

# Plot Predictions vs Actuals
def plot_predictions(y_true, y_pred, model_name):
    plt.figure(figsize=(8, 5))
    plt.scatter(range(len(y_true)), y_true, color='blue', label='Actual')
    plt.scatter(range(len(y_pred)), y_pred, color='red', label='Predicted')
    plt.title(f'{model_name} Predictions vs Actual')
    plt.xlabel('Sample Index')
    plt.ylabel('CO₂ Emissions')
    plt.legend()
    plt.grid()
    plt.show()

# Plot Residuals Histogram
def plot_residuals(y_true, y_pred, model_name):
    residuals = y_true - y_pred
    plt.figure(figsize=(8, 5))
    plt.hist(residuals, bins=30, color='purple', alpha=0.7)
    plt.title(f'{model_name} Residuals Histogram')
    plt.xlabel('Residual')
    plt.ylabel('Frequency')
    plt.grid()
    plt.show()

# Plot Correlation Heatmap
def plot_heatmap(y_true, predictions, model_names):
    correlation_matrix = pd.DataFrame(predictions, columns=model_names).corr()
    plt.figure(figsize=(8, 6))
    sns.heatmap(correlation_matrix, annot=True, cmap='coolwarm', fmt=".2f")
    plt.title('Correlation Heatmap of Predictions Across Models')
    plt.show()

# Train and Evaluate Each Model
models = {}
predictions = {}

# LSTM Model
lstm_model = Sequential([
    LSTM(64, activation='relu', input_shape=(1, X_train_lstm.shape[2]), kernel_regularizer=l2(0.01)),
    Dropout(0.2),
    Dense(1)
])
lstm_model.compile(optimizer='adam', loss='mse')
history = lstm_model.fit(X_train_lstm, y_train, epochs=10, batch_size=128, validation_split=0.2, 
                         callbacks=[EarlyStopping(monitor='val_loss', patience=5, restore_best_weights=True)])
# Save Training History to CSV
history_df = pd.DataFrame(history.history)  # Convert training history to a DataFrame
history_csv_path = os.path.join(base_dir, "lstm_training_history.csv")  # Specify the path for saving the CSV
history_df.to_csv(history_csv_path, index=False)  # Save the CSV
print(f"Saved training history to {history_csv_path}")  # Confirm the file was saved

models['LSTM'] = lstm_model
lstm_model_path = os.path.join(model_save_dir, "lstm_model.h5")
lstm_model.save(lstm_model_path)
print(f"Saved LSTM model to {lstm_model_path}")
plot_training_history(history, "LSTM")
predictions['LSTM'] = lstm_model.predict(X_test_lstm)

# Save LSTM Evaluation Results to CSV
y_pred = lstm_model.predict(X_test_lstm).flatten()  # Generate predictions
mae, rmse, r2 = calculate_metrics(y_test.flatten(), y_pred)  # Predefined function for MAE, RMSE, and R²
mse, mape, smape = calculate_additional_metrics(y_test.flatten(), y_pred)  # Additional metrics
evaluation_results_path = os.path.join(base_dir, "lstm_evaluation_results.csv")  # File specific to LSTM
with open(evaluation_results_path, "w") as f:
    f.write("Metric,Value\n")
    f.write(f"MSE,{mse}\n")
    f.write(f"RMSE,{rmse}\n")
    f.write(f"MAE,{mae}\n")
    f.write(f"MAPE (%),{mape}\n")
    f.write(f"SMAPE (%),{smape}\n")
    f.write(f"R²,{r2}\n")
print(f"Saved LSTM evaluation results to {evaluation_results_path}")


# RNN Model
rnn_model = Sequential([
    SimpleRNN(64, activation='relu', input_shape=(1, X_train_lstm.shape[2]), kernel_regularizer=l2(0.01)),
    Dropout(0.2),
    Dense(1)
])
rnn_model.compile(optimizer='adam', loss='mse')
history = rnn_model.fit(X_train_lstm, y_train, epochs=10, batch_size=128, validation_split=0.2, 
                         callbacks=[EarlyStopping(monitor='val_loss', patience=5, restore_best_weights=True)])
# Save RNN Training History to CSV
history_df = pd.DataFrame(history.history)  # Convert training history to a DataFrame
history_csv_path = os.path.join(base_dir, "rnn_training_history.csv")  # File name specific to RNN
history_df.to_csv(history_csv_path, index=False)  # Save the CSV
print(f"Saved RNN training history to {history_csv_path}")  # Confirm the file was saved

models['RNN'] = rnn_model
rnn_model.save('rnn_model.h5')
plot_training_history(history, "RNN")
predictions['RNN'] = rnn_model.predict(X_test_lstm)

# Save RNN Evaluation Results to CSV
y_pred = rnn_model.predict(X_test_lstm).flatten()  # Generate predictions
mae, rmse, r2 = calculate_metrics(y_test.flatten(), y_pred)
mse, mape, smape = calculate_additional_metrics(y_test.flatten(), y_pred)
evaluation_results_path = os.path.join(base_dir, "rnn_evaluation_results.csv")  # File specific to RNN
with open(evaluation_results_path, "w") as f:
    f.write("Metric,Value\n")
    f.write(f"MSE,{mse}\n")
    f.write(f"RMSE,{rmse}\n")
    f.write(f"MAE,{mae}\n")
    f.write(f"MAPE (%),{mape}\n")
    f.write(f"SMAPE (%),{smape}\n")
    f.write(f"R²,{r2}\n")
print(f"Saved RNN evaluation results to {evaluation_results_path}")


# GRU Model
gru_model = Sequential([
    GRU(64, activation='relu', input_shape=(1, X_train_lstm.shape[2]), kernel_regularizer=l2(0.01)),
    Dropout(0.2),
    Dense(1)
])
gru_model.compile(optimizer='adam', loss='mse')
history = gru_model.fit(X_train_lstm, y_train, epochs=10, batch_size=128, validation_split=0.2, 
                         callbacks=[EarlyStopping(monitor='val_loss', patience=5, restore_best_weights=True)])
# Save GRU Training History to CSV
history_df = pd.DataFrame(history.history)  # Convert training history to a DataFrame
history_csv_path = os.path.join(base_dir, "gru_training_history.csv")  # File name specific to GRU
history_df.to_csv(history_csv_path, index=False)  # Save the CSV
print(f"Saved GRU training history to {history_csv_path}")  # Confirm the file was saved

models['GRU'] = gru_model
gru_model.save('gru_model.h5')
plot_training_history(history, "GRU")
predictions['GRU'] = gru_model.predict(X_test_lstm)

# Save GRU Evaluation Results to CSV
y_pred = gru_model.predict(X_test_lstm).flatten()  # Generate predictions
mae, rmse, r2 = calculate_metrics(y_test.flatten(), y_pred)
mse, mape, smape = calculate_additional_metrics(y_test.flatten(), y_pred)
evaluation_results_path = os.path.join(base_dir, "gru_evaluation_results.csv")  # File specific to GRU
with open(evaluation_results_path, "w") as f:
    f.write("Metric,Value\n")
    f.write(f"MSE,{mse}\n")
    f.write(f"RMSE,{rmse}\n")
    f.write(f"MAE,{mae}\n")
    f.write(f"MAPE (%),{mape}\n")
    f.write(f"SMAPE (%),{smape}\n")
    f.write(f"R²,{r2}\n")
print(f"Saved GRU evaluation results to {evaluation_results_path}")


# Regression Model (Dense Network)
reg_model = Sequential([
    Dense(128, activation='relu', input_dim=X_train_preprocessed.shape[1], kernel_regularizer=l2(0.01)),
    Dropout(0.2),
    Dense(64, activation='relu', kernel_regularizer=l2(0.01)),
    Dense(1)
])
reg_model.compile(optimizer='adam', loss='mse')
history = reg_model.fit(X_train_preprocessed.toarray(), y_train, epochs=10, batch_size=32, validation_split=0.2, 
                         callbacks=[EarlyStopping(monitor='val_loss', patience=5, restore_best_weights=True)])
# Save Regression Training History to CSV
history_df = pd.DataFrame(history.history)  # Convert training history to a DataFrame
history_csv_path = os.path.join(base_dir, "regression_training_history.csv")  # File name specific to Regression
history_df.to_csv(history_csv_path, index=False)  # Save the CSV
print(f"Saved Regression training history to {history_csv_path}")  # Confirm the file was saved

models['Regression'] = reg_model
reg_model.save('reg_model.h5')
plot_training_history(history, "Regression")
predictions['Regression'] = reg_model.predict(X_test_preprocessed.toarray())

# Save Regression Evaluation Results to CSV
y_pred = reg_model.predict(X_test_preprocessed.toarray()).flatten()  # Generate predictions
mae, rmse, r2 = calculate_metrics(y_test.flatten(), y_pred)
mse, mape, smape = calculate_additional_metrics(y_test.flatten(), y_pred)
evaluation_results_path = os.path.join(base_dir, "regression_evaluation_results.csv")  # File specific to Regression
with open(evaluation_results_path, "w") as f:
    f.write("Metric,Value\n")
    f.write(f"MSE,{mse}\n")
    f.write(f"RMSE,{rmse}\n")
    f.write(f"MAE,{mae}\n")
    f.write(f"MAPE (%),{mape}\n")
    f.write(f"SMAPE (%),{smape}\n")
    f.write(f"R²,{r2}\n")
print(f"Saved Regression evaluation results to {evaluation_results_path}")


# Seq2Seq Model
def build_seq2seq_model(input_shape):
    encoder_inputs = Input(shape=input_shape)
    encoder = LSTM(64, activation='relu', return_state=True, kernel_regularizer=l2(0.01))
    _, state_h, state_c = encoder(encoder_inputs)
    encoder_states = [state_h, state_c]

    decoder_inputs = Input(shape=(1, input_shape[1]))
    decoder_lstm = LSTM(64, activation='relu', return_sequences=True, return_state=True, kernel_regularizer=l2(0.01))
    decoder_outputs, _, _ = decoder_lstm(decoder_inputs, initial_state=encoder_states)
    decoder_dense = Dense(1)
    decoder_outputs = decoder_dense(decoder_outputs)

    model = Model([encoder_inputs, decoder_inputs], decoder_outputs)
    return model

seq2seq_model = build_seq2seq_model((1, X_train_lstm.shape[2]))
seq2seq_model.compile(optimizer='adam', loss='mse')
decoder_input_train = np.zeros((X_train_lstm.shape[0], 1, X_train_lstm.shape[2]))
history = seq2seq_model.fit([X_train_lstm, decoder_input_train], y_train, epochs=10, batch_size=32, validation_split=0.2,
                            callbacks=[EarlyStopping(monitor='val_loss', patience=5, restore_best_weights=True)])
# Save Seq2Seq Training History to CSV
history_df = pd.DataFrame(history.history)  # Convert training history to a DataFrame
history_csv_path = os.path.join(base_dir, "seq2seq_training_history.csv")  # File name specific to Seq2Seq
history_df.to_csv(history_csv_path, index=False)  # Save the CSV
print(f"Saved Seq2Seq training history to {history_csv_path}")  # Confirm the file was saved

models['Seq2Seq'] = seq2seq_model
seq2seq_model.save('seq2seq_model.h5')
plot_training_history(history, "Seq2Seq")
predictions['Seq2Seq'] = seq2seq_model.predict([X_test_lstm, np.zeros((X_test_lstm.shape[0], 1, X_test_lstm.shape[2]))])

# Save Seq2Seq Evaluation Results to CSV
y_pred = seq2seq_model.predict([X_test_lstm, np.zeros((X_test_lstm.shape[0], 1, X_test_lstm.shape[2]))]).flatten()  # Generate predictions
mae, rmse, r2 = calculate_metrics(y_test.flatten(), y_pred)
mse, mape, smape = calculate_additional_metrics(y_test.flatten(), y_pred)
evaluation_results_path = os.path.join(base_dir, "seq2seq_evaluation_results.csv")  # File specific to Seq2Seq
with open(evaluation_results_path, "w") as f:
    f.write("Metric,Value\n")
    f.write(f"MSE,{mse}\n")
    f.write(f"RMSE,{rmse}\n")
    f.write(f"MAE,{mae}\n")
    f.write(f"MAPE (%),{mape}\n")
    f.write(f"SMAPE (%),{smape}\n")
    f.write(f"R²,{r2}\n")
print(f"Saved Seq2Seq evaluation results to {evaluation_results_path}")


# Transformer Model
def build_transformer_model(input_dim, seq_length):
    inputs = Input(shape=(seq_length, input_dim))
    attention = MultiHeadAttention(num_heads=4, key_dim=32)(inputs, inputs)
    attention = Dropout(0.1)(attention)
    attention = LayerNormalization(epsilon=1e-6)(attention + inputs)

    outputs = Dense(1)(Flatten()(attention))
    model = Model(inputs, outputs)
    return model

transformer_model = build_transformer_model(X_train_lstm.shape[2], 1)
transformer_model.compile(optimizer='adam', loss='mse')
history = transformer_model.fit(X_train_lstm, y_train, epochs=10, batch_size=62, validation_split=0.2,
                                callbacks=[EarlyStopping(monitor='val_loss', patience=5, restore_best_weights=True)])
# Save Transformer Training History to CSV
history_df = pd.DataFrame(history.history)  # Convert training history to a DataFrame
history_csv_path = os.path.join(base_dir, "transformer_training_history.csv")  # File name specific to Transformer
history_df.to_csv(history_csv_path, index=False)  # Save the CSV
print(f"Saved Transformer training history to {history_csv_path}")  # Confirm the file was saved

models['Transformer'] = transformer_model
transformer_model.save('transformer_model.h5')
plot_training_history(history, "Transformer")
predictions['Transformer'] = transformer_model.predict(X_test_lstm)

# Save Transformer Evaluation Results to CSV
y_pred = transformer_model.predict(X_test_lstm).flatten()  # Generate predictions
mae, rmse, r2 = calculate_metrics(y_test.flatten(), y_pred)
mse, mape, smape = calculate_additional_metrics(y_test.flatten(), y_pred)
evaluation_results_path = os.path.join(base_dir, "transformer_evaluation_results.csv")  # File specific to Transformer
with open(evaluation_results_path, "w") as f:
    f.write("Metric,Value\n")
    f.write(f"MSE,{mse}\n")
    f.write(f"RMSE,{rmse}\n")
    f.write(f"MAE,{mae}\n")
    f.write(f"MAPE (%),{mape}\n")
    f.write(f"SMAPE (%),{smape}\n")
    f.write(f"R²,{r2}\n")
print(f"Saved Transformer evaluation results to {evaluation_results_path}")


# Evaluate Models on Test Data and Visualize Results
model_names = list(models.keys())
for model_name, model_preds in predictions.items():
    y_pred_flattened = model_preds.flatten()  # Flatten predictions for compatibility
    mae, rmse, r2 = calculate_metrics(y_test.flatten(), y_pred_flattened)
    print(f"{model_name} - MAE: {mae:.2f}, RMSE: {rmse:.2f}, R²: {r2:.2f}")
    plot_predictions(y_test.flatten(), y_pred_flattened, model_name)
    plot_residuals(y_test.flatten(), y_pred_flattened, model_name)

# Define Carbon Tax Rate (Taiwan) in NT$ per ton of CO₂
carbon_tax_rate_per_ton = 300  # Example rate, replace with the actual value if known

# Function to Calculate Carbon Tax
def calculate_carbon_tax(y_pred, tax_rate_per_ton):
    """
    Calculate the carbon tax for predicted CO₂ emissions.
    :param y_pred: Predicted CO₂ emissions (in grams per mile)
    :param tax_rate_per_ton: Carbon tax rate in NT$ per ton of CO₂
    :return: Carbon tax amount in NT$
    """
    # Convert emissions from grams/mile to tons/mile (1 ton = 1,000,000 grams)
    emissions_in_tons_per_mile = y_pred / 1_000_000
    # Calculate tax per mile
    tax_per_mile = emissions_in_tons_per_mile * tax_rate_per_ton
    return tax_per_mile

# Apply Carbon Tax Calculation for Each Model's Predictions
for model_name, model_preds in predictions.items():
    y_pred_flattened = model_preds.flatten()  # Flatten predictions for compatibility
    carbon_tax = calculate_carbon_tax(y_pred_flattened, carbon_tax_rate_per_ton)
    print(f"{model_name} - Predicted Carbon Tax (NT$ per mile):")
    print(carbon_tax[:10])  # Display the first 10 predicted tax values as an example

# Save Carbon Tax Predictions to CSV
carbon_tax_results = []
for model_name, model_preds in predictions.items():
    y_pred_flattened = model_preds.flatten()
    carbon_tax = calculate_carbon_tax(y_pred_flattened, carbon_tax_rate_per_ton)
    carbon_tax_results.append({
        "Model": model_name,
        "Predicted CO₂ Emissions (grams/mile)": y_pred_flattened,
        "Predicted Carbon Tax (NT$/mile)": carbon_tax
    })

# Convert carbon tax results into a DataFrame and save
carbon_tax_df = pd.DataFrame(carbon_tax_results)
carbon_tax_df.to_csv('predicted_carbon_tax.csv', index=False)
print("Carbon tax predictions saved to 'predicted_carbon_tax.csv'.")

# Visualization of Carbon Tax Predictions Across Models
for model_name, model_preds in predictions.items():
    y_pred_flattened = model_preds.flatten()
    carbon_tax = calculate_carbon_tax(y_pred_flattened, carbon_tax_rate_per_ton)
    plt.figure(figsize=(10, 6))
    plt.scatter(range(len(carbon_tax)), carbon_tax, color='green', label=f'Predicted Carbon Tax')
    plt.title(f'{model_name} - Predicted Carbon Tax (NT$ per mile)')
    plt.xlabel('Sample Index')
    plt.ylabel('Carbon Tax (NT$)')
    plt.legend()
    plt.grid()
    plt.savefig(os.path.join(charts_dir, f"{model_name}_carbon_tax.png"))  # Save the figure
    plt.show()


# Correlation Heatmap
flattened_predictions = [pred.flatten() for pred in predictions.values()]
plot_heatmap(y_test.flatten(), np.column_stack(flattened_predictions), model_names)

# Additional Metrics Function
def calculate_additional_metrics(y_true, y_pred):
    mape = np.mean(np.abs((y_true - y_pred) / y_true)) * 100
    smape = 100 * np.mean(2 * np.abs(y_true - y_pred) / (np.abs(y_true) + np.abs(y_pred)))
    mse = mean_squared_error(y_true, y_pred)
    return mse, mape, smape

# Collect Performance Metrics in a Table
results = []
for model_name, model_preds in predictions.items():
    mae, rmse, r2 = calculate_metrics(y_test.flatten(), model_preds.flatten())
    mse, mape, smape = calculate_additional_metrics(y_test.flatten(), model_preds.flatten())
    results.append({
        "Model": model_name,
        "MSE": mse,
        "RMSE": rmse,
        "MAE": mae,
        "MAPE (%)": mape,
        "SMAPE (%)": smape,
        "R²": r2
    })

# Convert results to a DataFrame and display
results_df = pd.DataFrame(results)
print("Overall Model Performance Metrics:")
print(results_df)

# Save results to a CSV for reference
results_df.to_csv('overall_model_performance.csv', index=False)

# Comparison Charts for Metrics Across Models
metrics = ['MSE', 'RMSE', 'MAE', 'MAPE (%)', 'SMAPE (%)', 'R²']
for metric in metrics:
    plt.figure(figsize=(10, 6))
    sns.barplot(x="Model", y=metric, data=results_df)
    plt.title(f'Model Comparison for {metric}')
    plt.ylabel(metric)
    plt.xlabel('Model')
    plt.xticks(rotation=45)
    plt.grid(axis='y')
    plt.show()

# Forecasting Period Analysis
forecast_results = []
forecast_periods = [30, 90, 180]  # Forecast periods in days

for period in forecast_periods:
    y_test_period = y_test[:period].flatten()
    for model_name, model_preds in predictions.items():
        y_pred_period = model_preds[:period].flatten()
        mae, rmse, r2 = calculate_metrics(y_test_period, y_pred_period)
        mse, mape, smape = calculate_additional_metrics(y_test_period, y_pred_period)
        forecast_results.append({
            "Period (Days)": period,
            "Model": model_name,
            "MSE": mse,
            "RMSE": rmse,
            "MAE": mae,
            "MAPE (%)": mape,
            "SMAPE (%)": smape,
            "R²": r2
        })

# Convert forecast results into a DataFrame and display
forecast_results_df = pd.DataFrame(forecast_results)
print("Forecast Performance Metrics Across Periods:")
print(forecast_results_df)

# Save forecast results to a CSV file
forecast_results_df.to_csv('forecast_performance.csv', index=False)

# Plot Training History as Line Charts
def plot_training_history(history, model_name):
    plt.figure(figsize=(8, 5))
    plt.plot(history.history['loss'], label='Training Loss')
    if 'val_loss' in history.history:
        plt.plot(history.history['val_loss'], label='Validation Loss')
    plt.title(f'{model_name} Training History')
    plt.xlabel('Epochs')
    plt.ylabel('Loss')
    plt.legend()
    plt.grid()
    save_path = os.path.join(charts_dir, f"{model_name}_training_history.png")
    plt.savefig(save_path)
    plt.close()
    print(f"Saved {model_name}_training_history.png to {save_path}")

# Plot Predictions vs Actuals
def plot_predictions(y_true, y_pred, model_name):
    y_true = y_true.flatten()
    y_pred = y_pred.flatten()
    plt.figure(figsize=(8, 5))
    plt.scatter(range(len(y_true)), y_true, color='blue', label='Actual')
    plt.scatter(range(len(y_pred)), y_pred, color='red', label='Predicted')
    plt.title(f'{model_name} Predictions vs Actual')
    plt.xlabel('Sample Index')
    plt.ylabel('CO₂ Emissions')
    plt.legend()
    plt.grid()
    plt.savefig(os.path.join(charts_dir, f"{model_name}_predictions_vs_actual.png"))
    plt.close()
    print(f"Saved {model_name}_predictions_vs_actual.png")

# Plot Residuals Histogram
def plot_residuals(y_true, y_pred, model_name):
    residuals = y_true.flatten() - y_pred.flatten()
    plt.figure(figsize=(8, 5))
    plt.hist(residuals, bins=30, color='purple', alpha=0.7)
    plt.title(f'{model_name} Residuals Histogram')
    plt.xlabel('Residual')
    plt.ylabel('Frequency')
    plt.grid()
    plt.savefig(os.path.join(charts_dir, f"{model_name}_residuals_histogram.png"))
    plt.close()
    print(f"Saved {model_name}_residuals_histogram.png")

# Plot Correlation Heatmap
def plot_heatmap(y_true, predictions, model_names):
    correlation_matrix = pd.DataFrame(predictions, columns=model_names).corr()
    plt.figure(figsize=(8, 6))
    sns.heatmap(correlation_matrix, annot=True, cmap='coolwarm', fmt=".2f")
    plt.title('Correlation Heatmap of Predictions Across Models')
    plt.savefig(os.path.join(charts_dir, "correlation_heatmap.png"))
    plt.close()
    print("Saved correlation_heatmap.png")

# Comparison Charts for Metrics Across Models
metrics = ['MSE', 'RMSE', 'MAE', 'MAPE (%)', 'SMAPE (%)', 'R²']
for metric in metrics:
    plt.figure(figsize=(10, 6))
    sns.barplot(x="Model", y=metric, data=results_df)
    plt.title(f'Model Comparison for {metric}')
    plt.ylabel(metric)
    plt.xlabel('Model')
    plt.xticks(rotation=45)
    plt.grid(axis='y')
    plt.savefig(os.path.join(charts_dir, f"{metric}_comparison.png"))  # Save figure
    plt.show()