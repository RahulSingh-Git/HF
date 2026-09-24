"""
================================================================================
🌟 Magic Picture Storyteller 📖: Colorful Kid-Friendly Version
================================================================================

This application processes an image, generates a 50-100 word child-friendly 
story, and converts it to audio, all within a vibrant, engaging UI suitable 
for kids aged 3-10.
"""

import io
import streamlit as st
from PIL import Image
from transformers import pipeline
from gtts import gTTS

# ==============================================================================
# SECTION A: CUSTOM KID-FRIENDLY UI (VIBRANT & EYE-CATCHY)
# ==============================================================================

def inject_custom_css():
    """Injects playful CSS to customize colors, fonts, and button styles."""
    st.markdown(
        """
        <style>
        /* Main background - light and friendly */
        .stApp {
            background-color: #f7faff;
        }

        /* Title styling - colorful and cartoonish */
        h1 {
            color: #ff6f61 !important;
            font-family: 'Comic Sans MS', cursive, sans-serif !important;
            font-weight: bold !important;
            text-shadow: 1px 1px 2px #ffc107;
            text-align: center;
        }

        /* Sidebar styling - softer background */
        .stSidebar {
            background-color: #f1f8ff !important;
        }

        /* Customize the primary button to be big and bright */
        .stButton>button {
            background-color: #4CAF50; /* Green */
            color: white;
            font-size: 20px;
            font-weight: bold;
            border-radius: 12px;
            border: 2px solid #3e8e41;
            padding: 10px 24px;
            transition: all 0.3s ease;
            width: 100%;
        }
        .stButton>button:hover {
            background-color: #45a049;
            box-shadow: 0 4px 8px 0 rgba(0,0,0,0.2);
        }

        /* Adjust padding for a cleaner look */
        .block-container {
            padding-top: 1rem;
            padding-bottom: 2rem;
        }
        </style>
        """,
        unsafe_allow_html=True
    )

def add_illustrations():
    """
    Adds absolute-positioned illustrations (Dinosaur, Barbie, Playground).
    YOU MUST REPLACE THE PLACEHOLDER URLs WITH YOUR HOSTED IMAGE LINKS.
    """
    st.markdown(
        f"""
        <!-- Example Illustrations (Placeholders) -->
        <div style="position: absolute; top: 10%; right: 5%; width: 15vw; z-index: 1;">
            <img src="https://path_to_your_hosted_dinosaur_image.png" style="width: 100%; opacity: 0.9;" alt="Dinosaur">
        </div>
        <div style="position: absolute; top: 15%; right: 25%; width: 12vw; z-index: 1;">
            <img src="https://path_to_your_hosted_barbie_image.png" style="width: 100%; opacity: 0.9;" alt="Barbie">
        </div>
        <div style="position: absolute; top: 60%; right: 10%; width: 25vw; z-index: 1;">
            <img src="https://path_to_your_hosted_playground_image.png" style="width: 100%; opacity: 0.8;" alt="Playground">
        </div>
        """,
        unsafe_allow_html=True
    )

# ==============================================================================
# SECTION B: AI MODEL INITIALIZATION & CACHING
# ==============================================================================

@st.cache_resource
def load_caption_model():
    """Loads the BLIP Image Captioning pipeline."""
    return pipeline(
        "image-to-text",
        model="Salesforce/blip-image-captioning-base"
    )

@st.cache_resource
def load_story_model():
    """Loads a lightweight text generation pipeline (GPT-2)."""
    return pipeline(
        "text-generation",
        model="gpt2"
    )

# ==============================================================================
# SECTION C: CORE MODULAR FUNCTIONS (LOGIC)
# ==============================================================================

def image_to_text(image: Image.Image, caption_pipeline) -> str:
    """Extracts caption text from a PIL image."""
    try:
        results = caption_pipeline(image)
        if results and len(results) > 0:
            return results[0]['generated_text']
        return "a colorful adventure"
    except Exception as err:
        st.error(f"Image Error: {err}")
        return "a magical scene"

def text_to_story(caption: str, story_pipeline) -> str:
    """Expands caption into a 50-100 word child-friendly story."""
    # Custom kid-friendly prompt
    prompt = f"Once upon a time, in a magical land full of smiles, there was {caption}. "
    
    try:
        output = story_pipeline(
            prompt,
            max_new_tokens=110,
            min_new_tokens=60,
            do_sample=True,
            top_k=50,
            temperature=0.8,
            pad_token_id=50256
        )
        full_story = output[0]['generated_text']
        
        # Basic cleanup: ensure it ends at a full sentence near target length
        sentences = full_story.replace('\n', ' ').split('. ')
        valid_sentences = []
        word_count = 0
        for s in sentences:
            clean_s = s.strip()
            if not clean_s: continue
            if not clean_s.endswith('.'): clean_s += '.'
            valid_sentences.append(clean_s)
            word_count += len(clean_s.split())
            if word_count >= 55: break # Target range structure
            
        story = " ".join(valid_sentences)
        if not story.endswith("The end!"):
            story += " And they all lived happily ever after. The end!"
        return story
    except Exception as err:
        st.error(f"Story Error: {err}")
        return "They went on a wonderful adventure and made many new friends!"

def story_to_speech(story_text: str) -> io.BytesIO:
    """Converts story text to an MP3 audio stream."""
    tts = gTTS(text=story_text, lang='en', slow=False)
    audio_buffer = io.BytesIO()
    tts.write_to_fp(audio_buffer)
    audio_buffer.seek(0)
    return audio_buffer

# ==============================================================================
# SECTION D: MAIN APPLICATION ORCHESTRATOR
# ==============================================================================

def main():
    """
    Main orchestrator for the kid-friendly Storyteller application.
    Handles UI layout, loading models, and processing data.
    """
    # 1. Page Config
    st.set_page_config(
        page_title="Magic Storyteller 🎨✨",
        page_icon="🦄",
        layout="centered"
    )

    # 2. Inject UI Enhancements
    inject_custom_css()
    # add_illustrations() # Uncomment if you have valid URLs hosted

    # 3. Main Title
    st.title("🌟 Magic Picture Storyteller 📖")
    st.write("Welcome, Little Explorer! Upload a picture to hear a magic story built just for you!")

    # 4. Sidebar Instructions
    with st.sidebar:
        st.header("🎈 How to Play")
        st.write("1. Upload a picture.")
        st.write("2. Press '🚀 Create a Story!'")
        st.write("3. Read and press Play!")
        st.info("💡 Designed for kids aged 3–10.", icon="💡")

    # 5. Load Models (with spinner)
    with st.spinner("Preparing the Magic Book... (Loading AI)"):
        caption_pipe = load_caption_model()
        story_pipe = load_story_model()

    # 6. Image Uploader
    uploaded_file = st.file_uploader(
        "Upload your picture here (JPG, PNG):",
        type=["jpg", "jpeg", "png"]
    )

    # 7. Main Processing Logic
    if uploaded_file is not None:
        try:
            image = Image.open(uploaded_file).convert("RGB")
            
            # Display Image Preview
            col1, col2 = st.columns([1, 1])
            with col1:
                st.subheader("🖼️ Your Picture")
                st.image(image, use_container_width=True)

            with col2:
                st.subheader("✨ Magic Story Time")
                # Main Interaction Button
                if st.button("🚀 Create a Story!"):
                    # Stage A: Generate Caption
                    with st.spinner("Analyzing your picture..."):
                        caption = image_to_text(image, caption_pipe)
                        st.caption(f"**AI Vision detected:** *\"{caption}\"*")

                    # Stage B: Generate Story
                    with st.spinner("Spinning a fairy tale..."):
                        story = text_to_story(caption, story_pipe)
                    st.success("Story Generated!")
                    st.markdown(f"> **{story}**")

                    # Validate Word Count (Compliance)
                    word_count = len(story.split())
                    st.caption(f"📏 Story Length: **{word_count} words**")

                    # Stage C: Generate Audio
                    with st.spinner("Creating audio voice..."):
                        audio_data = story_to_speech(story)
                        st.subheader("🔊 Listen to Your Story")
                        st.audio(audio_data, format="audio/mp3")

        except Exception as e:
            st.error(f"Error loading image: {e}")

# entry point
if __name__ == "__main__":
    main()
