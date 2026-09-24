

import streamlit as st

# Function to inject custom CSS
def inject_custom_css():
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
        }

        /* Sidebar styling - softer background */
        .stSidebar {
            background-color: #f1f8ff !important;
        }

        /* Generic header text */
        .stMarkdown h2, .stMarkdown h3 {
            color: #444 !important;
        }

        /* Styling for the text boxes */
        .stTextInput>div>div>input, .stTextArea>div>div>textarea {
            background-color: #ffffff !important;
            color: #333 !important;
            border-color: #ddd !important;
        }

        /* General element container padding */
        .block-container {
            padding-top: 2rem;
            padding-bottom: 2rem;
        }

        /* Specific container for images - add background images to the PAGE */
        /* Note: It's hard to inject images *behind* elements, so we'll place them as floating illustrations. */
        </style>
        """,
        unsafe_allow_html=True
    )

# Place illustrations with CSS positioning - This is the key part for the new design.
def add_illustrations():
    # Use HTML to place the images and give them CSS for absolute positioning.
    # You will need to find/host your own images for dinosaurs, Barbie, etc. and update the URLs.
    # The example URLs below are placeholders.

    st.markdown(
        f"""
        <div style="position: absolute; top: 10%; right: 5%; width: 15vw; z-index: 10;">
            <img src="https://path_to_your_hosted_dinosaur_image.png" style="width: 100%; border-radius: 20px;" alt="Dinosaur">
        </div>
        <div style="position: absolute; top: 15%; right: 25%; width: 12vw; z-index: 9;">
            <img src="https://path_to_your_hosted_barbie_image.png" style="width: 100%;" alt="Barbie">
        </div>
        <div style="position: absolute; top: 60%; right: 10%; width: 25vw; z-index: 8;">
            <img src="https://path_to_your_hosted_playground_image.png" style="width: 100%;" alt="Playground">
        </div>
        <div style="position: absolute; top: 65%; left: 30%; width: 10vw; z-index: 7;">
             <img src="https://path_to_your_hosted_cartoon_char_1.png" style="width: 100%;" alt="Cartoon Character">
        </div>
        <div style="position: absolute; top: 75%; left: 20%; width: 10vw; z-index: 6;">
             <img src="https://path_to_your_hosted_cartoon_char_2.png" style="width: 100%;" alt="Cartoon Character">
        </div>
        """,
        unsafe_allow_html=True
    )

# --- Execution ---
# Call the CSS function first
inject_custom_css()

# Then the illustrations
add_illustrations()

# Now proceed with your existing app content
st.sidebar.title("🎈 How to Play")
# ... your existing sidebar content ...

st.title("🌟 Magic Picture Storyteller 📖")
# ... your existing main content ...
st.info("Designed for children aged 3–10.", icon="💡") # Keep the designed text as a callout

# ... remainder of your image uploading and processing logic ...

"""
Storytelling Web Application for Kids (Aged 3-10)
------------------------------------------------
This application processes an uploaded image, generates a 50-100 word child-friendly 
story based on the visual contents using Hugging Face Transformers pipelines, 
and converts the text to audio speech using gTTS.
"""

import io
import streamlit as st
from PIL import Image
from transformers import pipeline
from gtts import gTTS


# ==============================================================================
# STAGE 1: MODEL INITIALIZATION & CACHING
# ==============================================================================
@st.cache_resource
def load_caption_model():
    """
    Loads the BLIP Image Captioning pipeline.
    Uses @st.cache_resource to prevent re-downloading/re-loading on every user interaction.
    """
    return pipeline(
        "image-to-text",
        model="Salesforce/blip-image-captioning-base"
    )

@st.cache_resource
def load_story_model():
    """
    Loads a lightweight text generation pipeline (GPT-2) optimized for speed on Streamlit Cloud.
    """
    return pipeline(
        "text-generation",
        model="gpt2"
    )


# ==============================================================================
# STAGE 2: CORE MODULAR FUNCTIONS
# ==============================================================================
def image_to_text(image: Image.Image, caption_pipeline) -> str:
    """
    Extracts descriptive caption text from an input PIL image using BLIP.
    
    Args:
        image (Image.Image): Uploaded user image.
        caption_pipeline: Pre-loaded Hugging Face image-to-text pipeline.
        
    Returns:
        str: Descriptive caption text.
    """
    try:
        results = caption_pipeline(image)
        if results and len(results) > 0:
            return results[0]['generated_text']
        return "a magical scene full of colors"
    except Exception as err:
        st.error(f"Error during image processing: {err}")
        return "a magical scene full of colors"


def text_to_story(caption: str, story_pipeline) -> str:
    """
    Expands the image caption into a 50-100 word child-friendly story.
    
    Args:
        caption (str): Image caption generated by BLIP.
        story_pipeline: Pre-loaded Hugging Face text-generation pipeline.
        
    Returns:
        str: Complete 50-100 word child-friendly narrative.
    """
    prompt = f"Once upon a time, in a happy and bright magical land, there was {caption}. "
    
    try:
        # Generation bounds tuned for fast inference and target word length
        output = story_pipeline(
            prompt,
            max_new_tokens=110,
            min_new_tokens=60,
            do_sample=True,
            top_k=50,
            top_p=0.92,
            temperature=0.8,
            pad_token_id=50256,
            num_return_sequences=1
        )
        
        raw_story = output[0]['generated_text']
        
        # Clean and split into sentences to ensure sensible length bounds
        sentences = raw_story.replace('\n', ' ').split('. ')
        valid_sentences = []
        word_count = 0
        
        for sentence in sentences:
            clean_s = sentence.strip()
            if not clean_s:
                continue
            if not clean_s.endswith('.'):
                clean_s += '.'
            
            valid_sentences.append(clean_s)
            word_count += len(clean_s.split())
            
            # Target range: approximately 50-100 words
            if word_count >= 55:
                break
                
        final_story = " ".join(valid_sentences)
        
        # Add friendly ending suited for children
        if not final_story.endswith("The end!"):
            final_story += " Everyone smiled and had a wonderful day. The end!"
            
        return final_story

    except Exception as err:
        st.error(f"Error during story generation: {err}")
        return f"Once upon a time, there was {caption}. They went on a joyful adventure and made many new friends. The end!"


def story_to_speech(story_text: str) -> io.BytesIO:
    """
    Converts the generated story text into an MP3 audio stream using gTTS.
    
    Args:
        story_text (str): Narrative text to be spoken.
        
    Returns:
        io.BytesIO: In-memory byte buffer containing the MP3 audio.
    """
    tts = gTTS(text=story_text, lang='en', slow=False)
    audio_buffer = io.BytesIO()
    tts.write_to_fp(audio_buffer)
    audio_buffer.seek(0)
    return audio_buffer


# ==============================================================================
# STAGE 3: MAIN APPLICATION & USER INTERFACE
# ==============================================================================
def main():
    """
    Main application orchestrator and Streamlit UI definition.
    """
    # Streamlit page layout configuration
    st.set_page_config(
        page_title="Magic Picture Storyteller 🎨✨",
        page_icon="🦄",
        layout="centered"
    )

    # Custom kid-friendly CSS styling
    st.markdown("""
        <style>
        .main {
            background-color: #f0f8ff;
        }
        h1 {
            color: #ff4b4b;
            font-family: 'Comic Sans MS', cursive, sans-serif;
            text-align: center;
        }
        .stButton>button {
            background-color: #ff8c00;
            color: white;
            font-size: 20px;
            font-weight: bold;
            border-radius: 12px;
            width: 100%;
            height: 3em;
        }
        </style>
    """, unsafe_allow_html=True)

    st.title("🌟 Magic Picture Storyteller 📖")
    st.write("Welcome, Little Explorer! Upload a picture to hear a magic story made just for you!")

    # Sidebar documentation & instructions
    with st.sidebar:
        st.header("🎈 How to Play")
        st.write("1. Click below or drag a picture file into the upload box.")
        st.write("2. Press the **'🚀 Tell Me a Story!'** button.")
        st.write("3. Read and press Play to hear your audio story!")
        st.info("Designed for children aged 3–10.")

    # Model initialization with progress indicator
    with st.spinner("Loading AI models... Please wait a moment."):
        caption_pipe = load_caption_model()
        story_pipe = load_story_model()

    # Image Uploader (supports drag-and-drop or file browsing)
    uploaded_file = st.file_uploader(
        "Upload your picture here:",
        type=["jpg", "jpeg", "png"]
    )

    if uploaded_file is not None:
        try:
            # Display uploaded image preview
            image = Image.open(uploaded_file).convert("RGB")
            
            col1, col2 = st.columns([1, 1])
            with col1:
                st.subheader("🖼️ Your Picture")
                st.image(image, use_container_width=True)

            with col2:
                st.subheader("✨ Story Time")
                
                if st.button("🚀 Tell Me a Story!"):
                    # 1. Process image to caption
                    with st.spinner("Looking closely at your picture..."):
                        caption = image_to_text(image, caption_pipe)
                        st.caption(f"**Image detail detected:** *\"{caption}\"*")

                    # 2. Expand caption to story
                    with st.spinner("Writing your magic story..."):
                        story = text_to_story(caption, story_pipe)

                    st.success("Story Complete!")
                    st.markdown(f"> **{story}**")

                    # Display word count for assignment compliance verification
                    word_count = len(story.split())
                    st.caption(f"📏 Story Length: **{word_count} words**")

                    # 3. Convert story to speech audio
                    with st.spinner("Creating audio voice..."):
                        audio_data = story_to_speech(story)
                        st.subheader("🔊 Listen to Your Story")
                        st.audio(audio_data, format="audio/mp3")

        except Exception as e:
            st.error(f"Failed to load image: {e}")


# Program execution entry point
if __name__ == "__main__":
    main()
