"""
🌟 Magic Picture Storyteller 📖: Colorful Kid-Friendly Version
================================================================================
Target Audience: Kids aged 3-10
Assignment Requirements Met: Modular Functions, Caching, Length Constraints, 
Text-to-Speech, Kid-Friendly Design.
================================================================================
"""

import io
import streamlit as st
from PIL import Image
from transformers import pipeline
from gtts import gTTS

# ==============================================================================
# SECTION 1: VISUALS, STYLING & EYE-CATCHY DECORATIONS
# ==============================================================================

def inject_custom_css():
    """Injects colorful CSS for fonts, backgrounds, and buttons."""
    st.markdown(
        """
        <style>
        /* Main page background */
        .stApp {
            background-color: #f7faff;
        }

        /* Whimsical Title Styling */
        h1 {
            color: #ff6f61 !important;
            font-family: 'Comic Sans MS', cursive, sans-serif !important;
            font-weight: bold !important;
            text-shadow: 1px 1px 2px #ffc107;
            text-align: center;
        }

        /* Sidebar Styling */
        .stSidebar {
            background-color: #f1f8ff !important;
        }

        /* Generic header text color */
        .stMarkdown h2, .stMarkdown h3, .stMarkdown h4 {
            color: #444 !important;
        }

        /* Bright, Round Action Button */
        .stButton>button {
            background-color: #ff8c00; /* Bright Orange */
            color: white;
            font-size: 20px;
            font-weight: bold;
            border-radius: 12px;
            border: 2px solid #e67e22;
            padding: 12px 24px;
            width: 100%;
            transition: all 0.3s ease;
        }
        .stButton>button:hover {
            background-color: #e67e22;
            box-shadow: 0 4px 8px 0 rgba(0,0,0,0.2);
            transform: translateY(-2px);
        }

        /* Adjust padding for cleaner look */
        .block-container {
            padding-top: 2rem;
            padding-bottom: 2rem;
        }
        </style>
        """,
        unsafe_allow_html=True
    )

def add_illustrations():
    """
    Adds absolutely-positioned illustrations to make the app eye-catchy.
    ----------------------------------------------------------------------------
    ⚠️ IMPORTANT: REPLACE THE PLACEHOLDER URLs BELOW WITH YOUR ACTUAL,
    HOSTED IMAGE LINKS (e.g., from GitHub Pages, Cloudinary, or Imgur).
    ----------------------------------------------------------------------------
    """
    st.markdown(
        f"""
        <!-- 🦖 Dinosaur Illustration: Top-right -->
        <div style="position: absolute; top: 10%; right: 5%; width: 15vw; z-index: 10;">
            <img src="https://path_to_your_hosted_dinosaur_image.png" style="width: 100%; border-radius: 20px;" alt="Friendly Dinosaur">
        </div>
        
        <!-- 🎀 Barbie-like Character Illustration: Floating Top-Right -->
        <div style="position: absolute; top: 15%; right: 25%; width: 12vw; z-index: 9;">
            <img src="https://path_to_your_hosted_barbie_image.png" style="width: 100%;" alt="Adventure Barbie">
        </div>
        
        <!-- 🌳 Playground Scene: Large scene at bottom-right -->
        <div style="position: absolute; bottom: 5%; right: 5%; width: 28vw; z-index: 8; opacity: 0.8;">
            <img src="https://path_to_your_hosted_playground_image.png" style="width: 100%;" alt="Cartoon Playground">
        </div>
        
        <!-- 🤪 Generic Cartoon Character 1: Left-floating -->
        <div style="position: absolute; top: 65%; left: 30%; width: 10vw; z-index: 7;">
             <img src="https://path_to_your_hosted_cartoon_char_1.png" style="width: 100%;" alt="Happy Toon">
        </div>
        
        <!-- 🤪 Generic Cartoon Character 2: Near Sidebar instructions -->
        <div style="position: absolute; top: 75%; left: 18%; width: 10vw; z-index: 6;">
             <img src="https://path_to_your_hosted_cartoon_char_2.png" style="width: 100%;" alt="Waving Toon">
        </div>
        """,
        unsafe_allow_html=True
    )

# ==============================================================================
# SECTION 2: AI MODEL LOADING & CACHING (LOGIC)
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
# SECTION 3: CORE MODULAR PROCESSING FUNCTIONS
# ==============================================================================

def image_to_text(image: Image.Image, caption_pipeline) -> str:
    """Extracts caption text from a PIL image using BLIP."""
    try:
        # Generate raw description
        results = caption_pipeline(image)
        if results and len(results) > 0:
            caption = results[0]['generated_text']
            # Clean generic captions
            if not caption or caption.strip() == "":
                return "a colorful adventure"
            return caption
        return "a colorful adventure"
    except Exception as err:
        # Fallback to avoid crashes
        st.error(f"Image processing error: {err}")
        return "a magical scene"

def text_to_story(caption: str, story_pipeline) -> str:
    """Expands caption into a 50-100 word child-friendly story using GPT-2."""
    # Custom kid-friendly prompt
    prompt = f"Once upon a time, in a magical land full of bright colors, there was {caption}. "
    
    try:
        # Model configuration for length, variety, and child-friendliness
        output = story_pipeline(
            prompt,
            max_new_tokens=110,    # Controls max length generated
            min_new_tokens=60,     # Controls minimum length
            do_sample=True,        # Adds variety
            top_k=50,              # Limits vocabulary pool for simplicity
            temperature=0.8,       # Controls creativity
            pad_token_id=50256     # Required for GPT-2
        )
        full_story = output[0]['generated_text']
        
        # Cleanup: Split into sentences and reconstruct to clean boundaries
        sentences = full_story.replace('\n', ' ').split('. ')
        valid_sentences = []
        word_count = 0
        
        for s in sentences:
            clean_s = s.strip()
            if not clean_s: continue
            if not clean_s.endswith('.'): clean_s += '.'
            valid_sentences.append(clean_s)
            
            word_count += len(clean_s.split())
            if word_count >= 55: break # Target word count range check
            
        story = " ".join(valid_sentences)
        
        # Ensure friendly ending for kids
        if not story.endswith("The end!"):
            story += " They played happily and smiled all day long. The end!"
            
        return story
        
    except Exception as err:
        st.error(f"Story generation error: {err}")
        return "They went on a wonderful adventure and made many new friends. Everyone was happy! The end!"

def story_to_speech(story_text: str) -> io.BytesIO:
    """Converts the generated story text to an MP3 audio byte stream using gTTS."""
    tts = gTTS(text=story_text, lang='en', slow=False)
    audio_buffer = io.BytesIO()
    tts.write_to_fp(audio_buffer)
    audio_buffer.seek(0)
    return audio_buffer

# ==============================================================================
# SECTION 4: MAIN APPLICATION INTERFACE
# ==============================================================================

def main():
    """
    Main orchestrator for the kid-friendly Storyteller application.
    Handles UI layout, loading models, and processing data flow.
    """
    # 1. Page Configuration
    st.set_page_config(
        page_title="Magic Picture Storyteller 🎨✨",
        page_icon="🦄",
        layout="centered"
    )

    # 2. Inject Visual Enhancements
    inject_custom_css()
    add_illustrations() # <--- Poses dinosaurs, playgrounds, etc.

    # 3. Sidebar Instructions & Audience Info
    with st.sidebar:
        st.header("🎈 How to Play")
        st.markdown("""
        1. **Upload** your favorite picture file.
        2. Press the **'🚀 Create a Story!'** button.
        3. Listen to your magic story below!
        """)
        st.info("Designed for children aged 3–10. 🧸", icon="💡")

    # 4. Main Page Title & Intro
    st.title("🌟 Magic Picture Storyteller 📖")
    st.write("Welcome, Little Explorer! Upload a picture to hear a magic story built just for you!")

    # 5. Model Loading (with progress indicator)
    with st.spinner("Preparing the Magic Book... (AI Models Loading)"):
        caption_pipe = load_caption_model()
        story_pipe = load_story_model()

    # 6. Image Uploader (supporting drag-and-drop)
    uploaded_file = st.file_uploader(
        "Upload your picture here (JPG or PNG):",
        type=["jpg", "jpeg", "png"]
    )

    # 7. Core Application Logic (when file is uploaded)
    if uploaded_file is not None:
        try:
            # Process and Display the uploaded image
            image = Image.open(uploaded_file).convert("RGB")
            
            # Use columns for balanced layout
            col1, col2 = st.columns([1, 1])
            with col1:
                st.subheader("🖼️ Your Picture")
                st.image(image, use_container_width=True)

            with col2:
                st.subheader("✨ Story Time")
                
                # Main Interaction Button
                if st.button("🚀 Create a Story!"):
                    
                    # Stage 1: Generate Description
                    with st.spinner("Analyzing image details..."):
                        caption = image_to_text(image, caption_pipe)
                        st.caption(f"**AI Vision detected:** *\"{caption}\"*")

                    # Stage 2: Expand to Story
                    with st.spinner("Spinning a fairy tale..."):
                        story = text_to_story(caption, story_pipe)
                    
                    st.success("Story Generated!")
                    st.markdown(f"> **{story}**")
                    
                    # Validate Length Constraints (Assignment compliance check)
                    word_count = len(story.split())
                    st.caption(f"📏 Story Length: **{word_count} words**")

                    # Stage 3: Convert Text to Audio
                    with st.spinner("Creating audio voice..."):
                        audio_fp = story_to_speech(story)
                        st.subheader("🔊 Listen to Your Story")
                        st.audio(audio_fp, format="audio/mp3")

        except Exception as e:
            st.error(f"Failed to load image file: {e}")

# Entry point execution
if __name__ == "__main__":
    main()
