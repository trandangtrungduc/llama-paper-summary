import gradio as gr
from llm_helper import generate_summary, generate_description

with gr.Blocks() as demo:
    with gr.Row():
        gr.Markdown(
                """
                # <p style="text-align:center;"> 🪿 Summarize Paper 🪿 </p>
                """)
        
    with gr.Row():
        with gr.Column(scale=1):
            title = gr.Textbox(label="🌸 Title of paper", placeholder="Please provide paper name, select model and information to extract before generating summary report.", lines=2)
            model = gr.Radio(["LLama 3.1 - 8B", "LLama 3.1 - 70B", "LLama 3.1 - 405B", "LLama 3.2 - 90B"], label="🚀 Model", info="Which model do you want to use?")
        with gr.Column(scale=0.5):
            inform = gr.CheckboxGroup(["Introduction", "Related Work", "Method", "Experiment"], label="📝 Information", info="What information do you want to extract?")
            temperature = gr.Slider(value=0, minimum=0, maximum=1, step=0.1, label="♨️ Temperature")
    gen_sum_btn = gr.Button("Generate summary")
    
    with gr.Row():
        with gr.Column(scale=2, min_width=300):
            output = gr.Textbox(label="🌹 Output", placeholder="Show summary here !!!", lines=20)
        with gr.Column(scale=1, min_width=300):
            author = gr.Textbox(label="👷 Author", lines=5)
            year = gr.Textbox(label="📅 Year", lines=1)
            venue = gr.Textbox(label="🎥 Venue", lines=1)
            github = gr.Textbox(label="💻 Github", lines=2)
            
    with gr.Row():
        gr.Markdown(
                """
                # <p style="text-align:center;"> 🪿 Vision Language - Llama 3.2 - 90B 🪿 </p>
                """)
    with gr.Row():
        with gr.Column(scale=0.5):
            img_url = gr.Textbox(label="🔗 Image URL", placeholder="Please provide the image link.", lines=1)
            gen_des_btn = gr.Button("Generate description")
        with gr.Column(scale=0.5):
            img_prompt = gr.Textbox(label="❓ Image Prompt", placeholder="Please provide the prompt.", lines=4)
    with gr.Row():
        with gr.Column(scale=0.5):
            img_output = gr.Image(label="🌿 Show image")
        with gr.Column(scale=0.5):
            des_output = gr.Textbox(label="⭐ Output", placeholder="Show description here !!!", lines=20)
                   
    gen_sum_btn.click(generate_summary, inputs=[title, model, inform, temperature], outputs=[output, author, year, venue, github]) 
    gen_des_btn.click(generate_description, inputs=[img_prompt, img_url], outputs=[img_output, des_output]) 
    
if __name__ == "__main__":
    demo.launch() 