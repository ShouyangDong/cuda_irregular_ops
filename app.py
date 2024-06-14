import gradio as gr
def transform(source_code, transformation):
    if transformation == "Loop Split":
        return "Welcome to Gradio, start to split for loop!"
    elif transformation == "loop fusion":
        return "Welcome to Gradio, start to fuse for loop!"
    elif transformation == "Loop Reorder":
        return "Welcome to Gradio, start to reorder for loop!"
    else:
        return "Welcome to Gradio, start to Recovery for loop!"


title = """<h1 align="center">🔥Falcon: Transcompile Your Code Automatically🚀</h1>"""

with gr.Blocks() as demo:
    gr.HTML('<center> <img src="/file=falcon.png" style="width: 300px; height: 300px;">')
    gr.HTML(title)


    inp = gr.Textbox(placeholder="Enter the source code", label="Source code")
    with gr.Row():
        dropdown = gr.Dropdown(choices=["Loop Reorder", "Loop Split", "Loop Fusion", "Loop Recovery"], label="Select a loop transformation option")
        out = gr.Textbox(label="Target code")

    btn = gr.Button("Run")
    btn.click(fn=transform, inputs=[inp, dropdown], outputs=out)


demo.launch(allowed_paths=["./"])