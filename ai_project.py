import tkinter as tk
from tkinter import ttk
from ttkbootstrap import Style
from sklearn.linear_model import LinearRegression
from sklearn.model_selection import train_test_split
import pandas as pd

# Sample Dataset
data = {
    "Size (sq ft)": [800, 1000, 1200, 1500, 1800],
    "Number of Rooms": [2, 3, 3, 4, 4],
    "Price (in $10)": [200, 250, 300, 400, 450]
}

# Convert to DataFrame
df = pd.DataFrame(data)

# Features and Target
X = df[["Size (sq ft)", "Number of Rooms"]]
y = df["Price (in $10)"]

# Train Model
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
model = LinearRegression()
model.fit(X_train, y_train)

# Conversion rate (update this as necessary)
conversion_rate = 80  # 1 USD = 80 INR

# Prediction Function
def predict_price():
    try:
        # Get user inputs
        size = float(entry_size.get())
        rooms = int(entry_rooms.get())

        # Prepare the input as a DataFrame with feature names
        input_data = pd.DataFrame([[size, rooms]], columns=["Size (sq ft)", "Number of Rooms"])

        # Predict Price
        predicted_price_usd = model.predict(input_data)[0]

        # Convert to Rupees
        predicted_price_inr = predicted_price_usd * 10 * conversion_rate

        # Display Result
        result_label.config(text=f"Predicted Price: ₹{predicted_price_inr:.2f}", foreground="green")
    except ValueError:
        result_label.config(text="Please enter valid inputs.", foreground="red")

# Initialize Tkinter Window
root = tk.Tk()
root.title("House Price Prediction")
root.geometry("500x400")

# Apply ttkbootstrap styling
style = Style("cosmo")  # Choose from themes like 'morph', 'cosmo', 'flatly', etc.
style.configure("TLabel", font=("Helvetica", 12))
style.configure("TButton", font=("Helvetica", 12, "bold"))

# Header
header_label = ttk.Label(root, text="🏠 House Price Prediction 🏠", font=("Helvetica", 18, "bold"), anchor="center")
header_label.pack(pady=20)

# Input Fields Frame
frame = ttk.Frame(root, padding=10)
frame.pack(pady=10)

ttk.Label(frame, text="Size (sq ft):").grid(row=0, column=0, padx=10, pady=5, sticky="w")
entry_size = ttk.Entry(frame, width=25)
entry_size.grid(row=0, column=1, padx=10, pady=5)

ttk.Label(frame, text="Number of Rooms:").grid(row=1, column=0, padx=10, pady=5, sticky="w")
entry_rooms = ttk.Entry(frame, width=25)
entry_rooms.grid(row=1, column=1, padx=10, pady=5)

# Predict Button
predict_button = ttk.Button(root, text="🔮 Predict Price", command=predict_price, bootstyle="success")
predict_button.pack(pady=20)

# Result Display
result_label = ttk.Label(root, text="", font=("Helvetica", 14, "bold"), anchor="center")
result_label.pack(pady=10)

# Run the application
root.mainloop()