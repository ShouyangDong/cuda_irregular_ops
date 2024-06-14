import gradio as gr
def loop_transform(source_code, transformation):
    if transformation == "Loop Split":
        return "Welcome to Gradio, start to split for loop!"
    elif transformation == "loop fusion":
        return "Welcome to Gradio, start to fuse for loop!"
    elif transformation == "Loop Reorder":
        return "Welcome to Gradio, start to reorder for loop!"
    else:
        return "Welcome to Gradio, start to Recovery for loop!"

def memory_conversion(source_code, transformation):
    if transformation=="Cache Read":
        return "Welcome to Gradio, start to Cache Read!"
    elif transformation=="Cache Write":
        return "Welcome to Gradio, start to Cache Write!"
    elif transformation=="Tensor Contraction":
        return "Welcome to Gradio, start to Tensor Contraction!"

def intrinsic_conversion(source_code, transformation):
    if transformation == "Tensorization":
        return "Welcome to Gradio, start to Tensorization!"
    if transformation == "Detensorization":
        return "Welcome to Gradio, start to Detensorization!"

title = """<h1 align="center">🔥Falcon: Transcompile Your Code Automatically🚀</h1>"""

with gr.Blocks() as demo:
    gr.HTML('<center> <img src="/file=falcon.png" style="width: 150px; height: 150px;">')
    gr.HTML(title)

    with gr.Row():
        
        source_platform = gr.Dropdown(choices=["Intel DL Boost", "NVIDIA GPU", "AMD MI", "Cambricon MLU"], label="Select source platform")
        inp = gr.Textbox(placeholder="Enter the source code", label="Source code")
        target_platform = gr.Dropdown(choices=["Intel DL Boost", "NVIDIA GPU", "AMD MI", "Cambricon MLU"], label="Select target platform")


    gr.HTML("""<h1> Sequentialization/Parallelization </h1>""")
    with gr.Row():
        dropdown = gr.Dropdown(choices=["Loop Reorder", "Loop Split", "Loop Fusion", "Loop Recovery"], label="Select a loop transformation option")
        btn = gr.Button("Run")
        out = gr.Textbox(label="Target code")

    btn.click(fn=loop_transform, inputs=[inp, dropdown], outputs=out)

    gr.HTML("""<h1> Memory Conversion </h1>""")
    with gr.Row():
        dropdown = gr.Dropdown(choices=["Cache Read", "Cache Write", "Tensor Contraction"], label="Select a memory conversion option")
        mem_btn = gr.Button("Run")
        mem_out = gr.Textbox(label="Target code")
    
    mem_btn.click(fn=memory_conversion, inputs=[out, dropdown], outputs=mem_out)

    gr.HTML("""<h1> (De)tensorization </h1>""")
    with gr.Row():
        dropdown = gr.Dropdown(choices=["Tensorization", "Detensorization"], label="Select a tensorization option")
        intrin_btn = gr.Button("Run")
        intrin_out = gr.Textbox(label="Target code")
    
    intrin_btn.click(fn=intrinsic_conversion, inputs=[mem_out, dropdown], outputs=intrin_out)

demo.launch(allowed_paths=["./"])