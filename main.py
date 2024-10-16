import gradio as gr
from tensorflow.keras.preprocessing.text import Tokenizer
from tensorflow.keras.models import load_model
from tensorflow.keras.preprocessing.sequence import pad_sequences
model = load_model('model.keras')

word_tokenizer = Tokenizer()

def preprocess_input(text):
    MAX_LENGTH = 100
    sequence = word_tokenizer.texts_to_sequences([text])
    padded_sequence = pad_sequences(sequence, maxlen=MAX_LENGTH)
    return padded_sequence

def predict(text):
    processed_input = preprocess_input(text)
    prediction = model.predict(processed_input)
    return "Spam" if prediction > 0.5 else "Not Spam"

# Create a Gradio interface
gradio_app = gr.Interface(
    fn=predict,
    inputs="textbox",
    outputs="text",
    title="Spam Detection Model",
    description="Enter a message to check if it's spam or not."
)

# Launch Gradio using CLI
if __name__ == "__main__":
    gradio_app.launch()
