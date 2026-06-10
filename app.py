import streamlit as st
import numpy as np
import pickle
import random
from tensorflow.keras.models import load_model
from music21 import stream, note, chord

st.title("🎵 AI Music Generator")
st.write("Click the button to generate AI music!")

# Load model and notes
@st.cache_resource
def load_everything():
    model = load_model("music_model.h5")
    with open("notes.pkl", "rb") as f:
        notes = pickle.load(f)
    return model, notes

model, notes = load_everything()

unique_notes = sorted(set(notes))
note_to_int = {n: i for i, n in enumerate(unique_notes)}
int_to_note = {i: n for n, i in note_to_int.items()}
sequence_length = 50

def generate_music():
    start = random.randint(0, len(notes) - sequence_length - 1)
    pattern = [note_to_int[n] for n in notes[start:start+sequence_length]]
    generated = []

    for _ in range(200):
        inp = np.reshape(pattern, (1, len(pattern), 1)) / len(unique_notes)
        pred = model.predict(inp, verbose=0)
        index = np.argmax(pred)
        generated.append(int_to_note[index])
        pattern.append(index)
        pattern = pattern[1:]

    output = stream.Stream()
    for pattern in generated:
        if '.' in pattern:
            chord_notes = [note.Note(int(n)) for n in pattern.split('.')]
            new_chord = chord.Chord(chord_notes)
            output.append(new_chord)
        else:
            try:
                new_note = note.Note(pattern)
                new_note.duration.quarterLength = 0.5
                output.append(new_note)
            except:
                pass

    output.write('midi', fp="output.mid")
    return "output.mid"

if st.button("🎵 Generate Music"):
    with st.spinner("Generating music..."):
        output_file = generate_music()
        st.success("✅ Music Generated!")
        with open(output_file, "rb") as f:
            st.download_button(
                label="⬇️ Download MIDI",
                data=f,
                file_name="generated_music.mid",
                mime="audio/midi"
            )