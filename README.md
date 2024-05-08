# Simple Chatbot

This is a simple chatbot implemented in Python using TensorFlow/Keras. It can understand and respond to various intents based on the user's input.

## Getting Started

These instructions will get you a copy of the project up and running on your local machine for development and testing purposes.

### Prerequisites

- Python 3.6+
- pip (Python package manager)

### Installation

1. Clone the repository:

   ```bash
   git clone https://github.com/your-username/simple-chatbot.git
   ```

2. Navigate to the project directory:

   ```bash
   cd simple-chatbot
   ```

3. Install the dependencies using pip:

   ```bash
   pip install -r requirements.txt
   ```

## Usage

1. **Training the Chatbot**:

   - Make sure you have a `intends.json` file containing your training data.
   - Run the training script:

     ```bash
     python train_chatbot.py
     ```

   This will train the chatbot model and save it along with the tokenizer and label encoder.

2. **Start Chatting**:

   - Once the chatbot is trained, you can start chatting with it using the interactive chat script:

     ```bash
     python chat.py
     ```

   The chatbot will prompt you to enter a message, and it will respond based on the trained model.

## File Structure

- `intends.json`: Contains training data in JSON format.
- `train_chatbot.py`: Script to train the chatbot model.
- `chat.py`: Script for interactive chatting with the trained model.
- `requirements.txt`: List of Python dependencies for the project.

