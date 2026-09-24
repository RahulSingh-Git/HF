"""
================================================================================
A simple storytelling web application tailored for children aged 3–10.

Core Functional Requirements Met:
1. Modular Functions for each stage.
2. Image Processing & Captioning (Salesforce/blip-image-captioning-base).
3. Text-Generation (50-100 word narrative tailored for kids).
4. Text-to-Speech Conversion (gTTS for lightweight integration).
5. User Experience: Uses caching, meaningful names, and clear instructions.
================================================================================
"""

import io
import streamlit as st
from PIL import Image
from transformers import pipeline
from gtts import gTTS

# ==============================================================================
# SECTION A: CACHED MODEL LOADING FUNCTIONS
# ==============================================================================

@st.cache_resource
def load_caption_model():
    """Loads the Image Captioning pipeline."""
    return pipeline(
        "image-to-text",
        model="Salesforce/blip-image-captioning-base"
    )

@st.cache_resource
def load_story_model():
    """Loads a very lightweight Text Generation pipeline (GPT-2)."""
    return pipeline(
        "text-generation",
        model="gpt2"
    )


# ==============================================================================
# SECTION B: CORE MODULAR FUNCTIONS (Task Requirements)
# ==============================================================================

def process_and_caption_image(image: Image.Image, pipeline_object) -> str:
    """Requirement: Image Processing & Captioning."""
    try:
        results = pipeline_object(image)
        if results and len(results) > 0:
            # Clean up the output to be suitable for kids' prompts
            caption = results[0]['generated_text']
            # If model generates blank, use fallback
            if not caption or caption.strip() == "":
                return "a magical scene full of joy"
            return caption
        return "a magical scene"
    except Exception as e:
        st.error(f"Error analyzing image: {e}")
        return "a colorful adventure"

def generate_child_story(caption: str, pipeline_object) -> str:
    """
    Requirement: Story Generation (50-100 words tailored for kids).
    """
    # Create a simple, positive prompt tailored for kids 3-10
    prompt = f"Once upon a time, in a magical land full of bright colors, there was {caption}. "
    
    try:
        # Constraints tuned for gpt2 (CPU-friendly generation)
        response = pipeline_object(
            prompt,
            max_new_tokens=110, # Limits maximum generated tokens
            min_new_tokens=60,  # Limits minimum generated tokens
            do_sample=True,     # Introduces randomness/creativity
            top_k=50,           # Restricts vocabulary pool for simple language
            temperature=0.8,    # Increases creativity slightly
            pad_token_id=50256  # Necessary token identifier for GPT-2
        )
        full_text = response[0]['generated_text']
        
        # Cleanup logic: Split into sentences and reconstruct to ensure complete thoughts
        sentences = full_text.replace('\n', ' ').split('. ')
        valid_sentences = []
        word_count = 0
        
        for sentence in sentences:
            sentence_clean = sentence.strip()
            if not sentence_clean: continue
            
            # Ensure proper punctuation if missing
            if not sentence_clean.endswith('.'): sentence_clean += '.'
            
            valid_sentences.append(sentence_clean)
            word_count += len(sentence_clean.split())
            
            # Target range: ~50-100 words. Stop around 60 words to leave buffer.
            if word_count >= 60:
                break
                
        # Consolidate sentences into final narrative
        final_narrative = " ".join(valid_sentences)
        
        # Ensure standard child-friendly closing
        if not final_narrative.endswith("The end!"):
            final_narrative += " They played together happily all day long. The end!"
            
        return final_narrative

    except Exception as e:
        st.error(f"Error generating story: {e}")
        return f"Once upon a time, there was {caption}. They went on a joyful adventure, made many friends, and had a wonderful time! The end!"

def convert_story_to_speech(text: str) -> io.BytesIO:
    """Requirement: Text-to-Speech Conversion."""
    # slow=False creates standard speed; set to True if desired slower for kids 3-6.
    tts = gTTS(text=text, lang='en', slow=False)
    audio_buffer = io.BytesIO()
    tts.write_to_fp(audio_buffer)
    audio_buffer.seek(0)
    return audio_buffer


# ==============================================================================
# SECTION C: MAIN APPLICATION INTERFACE & EXECUTION
# ==============================================================================

def main():
    """Application Orchestrator."""
    # Basic kid-friendly app settings
    st.set_page_config(
        page_title="Magic Picture Storyteller 🎨✨",
        page_icon="🦄",
        layout="centered"
    )

    # Main Page Title
    st.title("🌟 Magic Picture Storyteller 📖")
    st.write("Welcome, Little Explorer! Upload a picture to hear a magic story built just for you!")

    # Sidebar documentation and instructions (Requirement: Suitable for users aged 3–10)
    with st.sidebar:
        st.header("🎮 How to Play")
        st.markdown("""
        1. Click 'Upload' to select your favorite picture.
        2. Press the **'🚀 Create a Story!'** button.
        3. Listen to your magic story below!
        """)
        st.info("Designed for kids aged 3–10. 🧸", icon="💡")

    # Pipeline initialization with progress indicator
    with st.spinner("Preparing the Magic Book... (AI Models Loading)"):
        caption_pipe = load_caption_model()
        story_pipe = load_story_model()

    # Image Uploader (supporting JPEG, JPG, and PNG)
    uploaded_file = st.file_uploader(
        "Upload your picture here (JPG or PNG):",
        type=["jpg", "jpeg", "png"]
    )

    if uploaded_file is not None:
        try:
            # display image preview
            image = Image.open(uploaded_file).convert("RGB")
            
            # layout columns for better display
            col1, col2 = st.columns([1, 1])
            with col1:
                st.subheader("🖼️ Your Picture")
                st.image(image, use_container_width=True)

            with col2:
                st.subheader("✨ Story Time")
                
                # Use standard big button for kids
                if st.button("🚀 Create a Story!"):
                    
                    # Stage 1: Generate Description
                    with st.spinner("Looking closely at picture details..."):
                        caption = process_and_caption_image(image, caption_pipe)
                        st.caption(f"**AI Vision detected:** *\"{caption}\"*")

                    # Stage 2: Expand to narrative
                    with st.spinner("Spinning a fairy tale..."):
                        story = generate_child_story(caption, story_pipe)
                    
                    # Requirement: Suitable for children
                    st.success("Story Generated!")
                    st.markdown(f"> **{story}**")
                    
                    # Show word count for compliance verification
                    word_count = len(story.split())
                    st.caption(f"📏 Narrative Length: **{word_count} words**")

                    # Stage 3: Convert Text to Audio
                    with st.spinner("Creating audio voice..."):
                        audio_fp = convert_story_to_speech(story)
                        # Requirement: Text-to-speech interaction
                        st.subheader("🔊 Listen to Your Story")
                        st.audio(audio_fp, format="audio/mp3")

        except Exception as e:
            st.error(f"Failed to process image file: {e}")

# entry point execution
if __name__ == "__main__":
    main()
